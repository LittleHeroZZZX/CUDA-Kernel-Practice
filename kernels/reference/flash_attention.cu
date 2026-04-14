#include <cuda_runtime.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Tile sizes.  d must equal MAX_D (standard for modern LLMs: 64 or 128).
//  d must also be a multiple of 4 (for float4 loads).
// ─────────────────────────────────────────────────────────────────────────────
#define BLOCK_Q 32
#define BLOCK_KV 32
#define MAX_D 128

// ─────────────────────────────────────────────────────────────────────────────
//  Flash Attention — float32, causal mask
//
//  Tensor layout : [B, H, N, d]  contiguous row-major
//  Grid           : (ceil(N / BLOCK_Q),  B * H)
//  Block          : BLOCK_Q threads;  thread t ↔ query row q_row
// ─────────────────────────────────────────────────────────────────────────────
__global__ void flash_attention_kernel(const float *__restrict__ Q,
                                       const float *__restrict__ K,
                                       const float *__restrict__ V,
                                       float *__restrict__ O, int N, int d,
                                       float scale) {
  const int bh = blockIdx.y;
  const int local_q = threadIdx.x;
  const int q_row = blockIdx.x * BLOCK_Q + local_q;

  // Base pointers for this (batch, head) slice.
  const int slice = bh * N * d;
  const float *Qbh = Q + slice;
  const float *Kbh = K + slice;
  const float *Vbh = V + slice;
  float *Obh = O + slice;

  // ─────────────────────────────────────────────────────────────────────
  //  Shared memory: one KV tile, 16 KB each, 32 KB total.
  //
  //  During LOAD  : accessed via flat pointer (Ks_flat)
  //  During GEMV  : accessed via 2-D indexing  Ks[kv][j]
  //  Both aliases point to the same storage — legal in CUDA.
  // ─────────────────────────────────────────────────────────────────────
  __shared__ float Ks[BLOCK_KV][MAX_D];
  __shared__ float Vs[BLOCK_KV][MAX_D];

  // ─────────────────────────────────────────────────────────────────────
  //  Load this thread's Q row into registers (done ONCE).
  //
  //  float4 = 128-bit load: 4 floats per instruction.
  //  Each thread issues d/4 = 32 vector loads.
  // ─────────────────────────────────────────────────────────────────────
  float q_reg[MAX_D] = {};
  if (q_row < N) {
    const float4 *Qrow4 = reinterpret_cast<const float4 *>(Qbh + q_row * d);
    for (int j4 = 0; j4 < d / 4; ++j4) {
      float4 v = Qrow4[j4];
      q_reg[j4 * 4 + 0] = v.x;
      q_reg[j4 * 4 + 1] = v.y;
      q_reg[j4 * 4 + 2] = v.z;
      q_reg[j4 * 4 + 3] = v.w;
    }
  }

  // Running state for online softmax.
  float m = -FLT_MAX;  // running max
  float l = 0.f;       // running normaliser
  float o[MAX_D] = {}; // output accumulator

  // ─────────────────────────────────────────────────────────────────────
  //  Main loop: iterate over KV tiles.
  // ─────────────────────────────────────────────────────────────────────
  for (int kv_start = 0; kv_start < N; kv_start += BLOCK_KV) {

    // ─────────────────────────────────────────────────────────────────
    //  Load K and V tiles — coalesced, conflict-free.
    //
    //  Tile = BLOCK_KV * d = 32 * 128 = 4096 floats.
    //  Assign round-robin: thread t owns indices t, t+32, t+64, …
    //
    //  Global memory: at step `iter`, thread t reads
    //    Kbh[kv_start*d + iter*32 + t]
    //    adjacent threads → adjacent addresses → ONE 128-byte transaction.
    //
    //  Shared memory: writes Ks_flat[iter*32 + t]
    //    → bank = (iter*32 + t) % 32 = t
    //    → every thread hits a different bank → ZERO conflicts.
    // ─────────────────────────────────────────────────────────────────
    float *Ks_flat = reinterpret_cast<float *>(Ks);
    float *Vs_flat = reinterpret_cast<float *>(Vs);
    const int tile_elems = BLOCK_KV * d; // 4096

    for (int s = local_q; s < tile_elems; s += BLOCK_Q) {
      const int row = s / d;
      const int col = s % d;
      const int global_row = kv_start + row;
      Ks_flat[row * MAX_D + col] = (global_row < N) ? Kbh[global_row * d + col] : 0.f;
      Vs_flat[row * MAX_D + col] = (global_row < N) ? Vbh[global_row * d + col] : 0.f;
    }
    __syncthreads();

    // ─────────────────────────────────────────────────────────────────
    //  Compute S[kv] = dot(q_reg, Ks[kv]) * scale,  causal mask.
    //
    //  All 32 threads read Ks[kv][j] simultaneously.
    //  Same address → SMEM broadcast (1 transaction per j). ✓
    //
    //  Causal mask: kv_start + kv > q_row → future positions.
    //  Once masked, all subsequent kv are also masked → break.
    // ─────────────────────────────────────────────────────────────────
    const int tile_len = min(BLOCK_KV, N - kv_start);

    float S[BLOCK_KV];
    for (int kv = 0; kv < BLOCK_KV; ++kv)
      S[kv] = -FLT_MAX; // default: masked / out-of-range

    if (q_row < N) {
      for (int kv = 0; kv < tile_len; ++kv) {
        if (kv_start + kv > q_row)
          break; // causal: rest are future

        float dot = 0.f;
        const float *Krow = Ks[kv];
        for (int j = 0; j < d; ++j)
          dot += q_reg[j] * Krow[j];
        S[kv] = dot * scale;
      }
    }

    // ─────────────────────────────────────────────────────────────────
    //  Online softmax update (Flash Attention Algorithm 1).
    //
    //  Invariant after each tile:
    //    o[j] = Σ exp(S[i] - m) * V[i][j]   (unnormalised)
    //    l    = Σ exp(S[i] - m)
    //  When m changes to m_new, multiply both by exp(m - m_new).
    // ─────────────────────────────────────────────────────────────────

    // Local max for this tile.
    float m_tile = -FLT_MAX;
    for (int kv = 0; kv < tile_len; ++kv)
      m_tile = fmaxf(m_tile, S[kv]);

    // New global max and rescale factor for previous state.
    const float m_new = fmaxf(m, m_tile);
    const float rescale = expf(m - m_new); // ≤ 1; = 1 if m unchanged

    // Rescale accumulated state to new baseline.
    l *= rescale;
    for (int j = 0; j < d; ++j)
      o[j] *= rescale;

    // Accumulate this tile.
    for (int kv = 0; kv < tile_len; ++kv) {
      const float e = expf(S[kv] - m_new); // ≈ 0 for masked slots
      const float *Vrow = Vs[kv];
      l += e;
      for (int j = 0; j < d; ++j)
        o[j] += e * Vrow[j];
    }
    m = m_new;

    __syncthreads(); // guard before next tile's SMEM write
  }

  // ─────────────────────────────────────────────────────────────────────
  //  Write output O = o / l.
  //
  //  Precompute inv_l to turn d divisions into d multiplications.
  //  float4 stores: 128-bit per instruction, coalesced.
  // ─────────────────────────────────────────────────────────────────────
  if (q_row < N) {
    const float inv_l = 1.f / l;
    float4 *Orow4 = reinterpret_cast<float4 *>(Obh + q_row * d);
    for (int j4 = 0; j4 < d / 4; ++j4) {
      Orow4[j4] = make_float4(o[j4 * 4 + 0] * inv_l, o[j4 * 4 + 1] * inv_l,
                              o[j4 * 4 + 2] * inv_l, o[j4 * 4 + 3] * inv_l);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-side launcher.
// ─────────────────────────────────────────────────────────────────────────────
void flash_attention(const float *Q, const float *K, const float *V, float *O,
                     int B, int H, int N, int d) {
  const float scale = 1.f / sqrtf((float)d);
  const dim3 grid((N + BLOCK_Q - 1) / BLOCK_Q, // Q tiles along sequence dim
                  B * H // one block-group per (batch, head)
  );
  const dim3 block(BLOCK_Q);
  flash_attention_kernel<<<grid, block>>>(Q, K, V, O, N, d, scale);
}