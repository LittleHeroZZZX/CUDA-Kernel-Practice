/*
 * ================================================================
 *  LLM Inference CUDA Kernels — Standard Reference Implementations
 * ================================================================
 *
 *  1. GEMM            — shared-memory tiling + register tiling
 *  2. Softmax         — row-wise, two-pass numerically stable
 *  3. LayerNorm       — two-pass (mean + var)
 *     RMSNorm         — single-pass sum-of-squares
 *  4. Block Reduce    — warp→block→grid, sum / max
 *  5. FlashAttention  — single-head, blocked online softmax
 *  7. Fused MHA       — naive single kernel (materializes score row)
 *
 *  Compile:  nvcc -O3 -arch=sm_80 llm_kernels.cu -o llm_kernels
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// ================================================================
//  Warp-level primitives  (reused by every kernel below)
// ================================================================

// Butterfly all-reduce within one warp (32 threads) — sum
// After this call every thread in the warp holds the same total.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// Butterfly all-reduce within one warp — max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

// Cross-warp block reduce using shared memory.
// Call after each warp has its partial result in `val`.
// Returns the block-wide result in thread 0; other threads are undefined.
__device__ float block_reduce_sum(float val) {
    __shared__ float smem[32];              // one slot per warp (max 32 warps)
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int nw  = (blockDim.x + 31) / 32;

    val = warp_reduce_sum(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();

    // First warp reduces smem[]
    val = (wid == 0 && lid < nw) ? smem[lid] : 0.f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float smem[32];
    int wid = threadIdx.x / 32;
    int lid = threadIdx.x % 32;
    int nw  = (blockDim.x + 31) / 32;

    val = warp_reduce_max(val);
    if (lid == 0) smem[wid] = val;
    __syncthreads();

    val = (wid == 0 && lid < nw) ? smem[lid] : -FLT_MAX;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

// Broadcast block result from thread 0 to all threads via smem.
// Must be called right after block_reduce_* while smem is still valid.
__device__ __forceinline__ float broadcast_from_zero(float val) {
    __shared__ float bcast;
    if (threadIdx.x == 0) bcast = val;
    __syncthreads();
    return bcast;
}


// ================================================================
//  1.  GEMM   C = A × B     A:[M,K]  B:[K,N]  C:[M,N]
// ================================================================
//
//  Two-level tiling strategy:
//
//    Block tile (TILE_M × TILE_N):
//      Each thread block owns one output tile of this size.
//      A-tile and B-tile are loaded into shared memory collaboratively,
//      reducing global memory traffic by a factor of TILE_K.
//
//    Register tile (WM × WN per thread):
//      Each thread owns a WM×WN sub-tile of the block tile in registers.
//      This increases arithmetic intensity and hides memory latency.
//
//  Arithmetic intensity:
//    Without tiling : 2×M×N×K FLOPs / (M×K + K×N + M×N) × 4B  ≈ O(1) FLOPs/B
//    With tiling    : increases by factor ~TILE_K → compute-bound regime
//
//  Launch:
//    grid  = (ceil(N/TILE_N), ceil(M/TILE_M))
//    block = (TILE_N/WN, TILE_M/WM) = (16, 16) → 256 threads

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WM     4       // each thread accumulates WM output rows
#define WN     4       // each thread accumulates WN output cols
// => blockDim = (TILE_N/WN, TILE_M/WM) = (16, 16)

__global__ void gemm_kernel(
    const float* __restrict__ A,   // [M, K] row-major
    const float* __restrict__ B,   // [K, N] row-major
    float*       __restrict__ C,   // [M, N] row-major
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];  // smem tile of A (64×16 × 4B = 4 KB)
    __shared__ float Bs[TILE_K][TILE_N];  // smem tile of B (16×64 × 4B = 4 KB)

    int tx  = threadIdx.x;                // thread col index within block
    int ty  = threadIdx.y;                // thread row index within block
    int tid = ty * blockDim.x + tx;       // linear thread id (0..255)
    int nt  = blockDim.x * blockDim.y;   // 256

    // Top-left global coordinates of this thread's WM×WN output
    int row0 = blockIdx.y * TILE_M + ty * WM;
    int col0 = blockIdx.x * TILE_N + tx * WN;

    // ---- Register accumulator (stays in registers throughout) ----
    float acc[WM][WN] = {};              // zero-initialize

    // ---- Main K-loop: stride through K dimension in TILE_K chunks ----
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // -- Load A[block_row : block_row+TILE_M, k0 : k0+TILE_K] into smem --
        // All 256 threads collaborate; each loads TILE_M*TILE_K/256 = 4 elements.
        for (int i = tid; i < TILE_M * TILE_K; i += nt) {
            int sm_r = i / TILE_K,  sm_c = i % TILE_K;
            int gr   = blockIdx.y * TILE_M + sm_r;
            int gc   = k0 + sm_c;
            // Bounds guard handles non-multiple-of-tile shapes
            As[sm_r][sm_c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }

        // -- Load B[k0 : k0+TILE_K, block_col : block_col+TILE_N] into smem --
        for (int i = tid; i < TILE_K * TILE_N; i += nt) {
            int sm_r = i / TILE_N,  sm_c = i % TILE_N;
            int gr   = k0 + sm_r;
            int gc   = blockIdx.x * TILE_N + sm_c;
            Bs[sm_r][sm_c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }

        __syncthreads();    // smem tiles are fully populated

        // -- Compute: outer product over the TILE_K inner dimension --
        // Registers a[], b[] avoid repeated smem reads inside the k-loop.
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a[WM], b[WN];
            #pragma unroll
            for (int m = 0; m < WM; ++m) a[m] = As[ty * WM + m][kk];
            #pragma unroll
            for (int n = 0; n < WN; ++n) b[n] = Bs[kk][tx * WN + n];
            // FMA: acc[m][n] += a[m] * b[n]
            #pragma unroll
            for (int m = 0; m < WM; ++m)
                #pragma unroll
                for (int n = 0; n < WN; ++n)
                    acc[m][n] += a[m] * b[n];
        }

        __syncthreads();    // required before next tile overwrites smem
    }

    // ---- Write register tile back to global memory ----
    #pragma unroll
    for (int m = 0; m < WM; ++m)
        #pragma unroll
        for (int n = 0; n < WN; ++n) {
            int r = row0 + m,  c = col0 + n;
            if (r < M && c < N) C[r * N + c] = acc[m][n];
        }
}
// Launch helper:
//   dim3 block(TILE_N/WN, TILE_M/WM);   // (16, 16)
//   dim3 grid(ceil(N/TILE_N), ceil(M/TILE_M));
//   gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);


// ================================================================
//  2.  Softmax  (row-wise, numerically stable two-pass)
// ================================================================
//
//  For each row i:
//    m_i   = max_j(x[i,j])                      ← pass 1: global max
//    s_i   = Σ_j exp(x[i,j] - m_i)             ← pass 2: shifted exp + sum
//    out[i,j] = exp(x[i,j] - m_i) / s_i        ← normalize
//
//  Subtracting the row max prevents overflow in exp() — standard trick.
//
//  One block per row.  blockDim.x = min(cols, 1024).

__global__ void softmax_kernel(
    const float* __restrict__ x,    // [rows, cols]
    float*       __restrict__ out,  // [rows, cols]
    int cols
) {
    int row = blockIdx.x;
    const float* rx = x   + (long)row * cols;
    float*       ro = out + (long)row * cols;
    int tid = threadIdx.x;

    // ---- Pass 1: row max ----
    float local_max = -FLT_MAX;
    for (int j = tid; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, rx[j]);

    float row_max = block_reduce_max(local_max);
    row_max = broadcast_from_zero(row_max);

    // ---- Pass 2: exp(x - max) and partial sum ----
    float local_sum = 0.f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float e = expf(rx[j] - row_max);
        ro[j]   = e;
        local_sum += e;
    }

    float row_sum = block_reduce_sum(local_sum);
    row_sum = broadcast_from_zero(row_sum);

    // ---- Pass 3: normalize ----
    for (int j = tid; j < cols; j += blockDim.x)
        ro[j] /= row_sum;
}
// Launch: softmax_kernel<<<rows, min(cols, 1024)>>>(x, out, cols);


// ================================================================
//  3a. LayerNorm
// ================================================================
//
//  out = (x - μ) / sqrt(σ² + ε) * γ + β
//
//  Two passes over the row: one for mean, one for variance.
//  Using Welford's online algorithm is an alternative (more numerically
//  stable for fp16) but the two-pass version is clearer here.
//
//  One block per token (row).  blockDim.x = min(hidden_dim, 1024).

__global__ void layernorm_kernel(
    const float* __restrict__ x,      // [rows, cols]
    const float* __restrict__ gamma,  // [cols]  learnable scale
    const float* __restrict__ beta,   // [cols]  learnable bias
    float*       __restrict__ out,    // [rows, cols]
    int cols, float eps
) {
    int row = blockIdx.x;
    const float* rx = x   + (long)row * cols;
    float*       ro = out + (long)row * cols;
    int tid = threadIdx.x;

    // ---- Pass 1: mean ----
    float local_sum = 0.f;
    for (int j = tid; j < cols; j += blockDim.x) local_sum += rx[j];
    float mean = block_reduce_sum(local_sum) / cols;
    mean = broadcast_from_zero(mean);

    // ---- Pass 2: variance  E[(x-μ)²] ----
    float local_var = 0.f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float d = rx[j] - mean;
        local_var += d * d;
    }
    float var    = block_reduce_sum(local_var) / cols;
    float inv_std = rsqrtf(broadcast_from_zero(var) + eps);  // 1/sqrt(σ²+ε)

    // ---- Normalize + affine ----
    for (int j = tid; j < cols; j += blockDim.x)
        ro[j] = (rx[j] - mean) * inv_std * gamma[j] + beta[j];
}

// ================================================================
//  3b. RMSNorm  (used in LLaMA / Qwen — no mean subtraction)
// ================================================================
//
//  out = x / rms(x) * γ,   rms(x) = sqrt(E[x²] + ε)
//
//  Cheaper than LayerNorm: single-pass, no mean subtraction.
//  Empirically matches LayerNorm quality when combined with pre-norm.

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,   // [cols]
    float*       __restrict__ out,
    int cols, float eps
) {
    int row = blockIdx.x;
    const float* rx = x   + (long)row * cols;
    float*       ro = out + (long)row * cols;
    int tid = threadIdx.x;

    // ---- Compute E[x²] ----
    float local_ss = 0.f;
    for (int j = tid; j < cols; j += blockDim.x)
        local_ss += rx[j] * rx[j];

    float rms    = block_reduce_sum(local_ss) / cols;
    float inv_rms = rsqrtf(broadcast_from_zero(rms) + eps);

    // ---- Scale ----
    for (int j = tid; j < cols; j += blockDim.x)
        ro[j] = rx[j] * inv_rms * gamma[j];
}
// Launch both norms: kernel<<<rows, min(cols, 1024)>>>(...);


// ================================================================
//  4.  Block Reduce  (standalone kernel, two-level grid pattern)
// ================================================================
//
//  For arrays that span many blocks we need a two-kernel pattern:
//
//    Kernel 1 (block_reduce_kernel):
//      Each block accumulates its own partial sum → partial[blockIdx.x].
//      Uses grid-stride loop so any N fits regardless of grid size.
//
//    Kernel 2 (final_reduce_kernel):
//      Launch with a single block to reduce partial[] → out[0].
//
//  Three-level hierarchy inside each block:
//    Thread level  → each thread loops over its assigned elements
//    Warp level    → __shfl_xor_sync butterfly (zero smem traffic)
//    Block level   → warp leaders write smem, warp-0 reduces smem

// ---- Kernel 1: partial reduce over large array ----
__global__ void block_reduce_sum_kernel(
    const float* __restrict__ in,      // [N]
    float*       __restrict__ partial, // [gridDim.x]
    int N
) {
    int tid = threadIdx.x;

    // Grid-stride accumulation: each thread covers multiple elements
    float sum = 0.f;
    for (int i = blockIdx.x * blockDim.x + tid;
             i < N;
             i += blockDim.x * gridDim.x)
        sum += in[i];

    // Block-wide reduce
    sum = block_reduce_sum(sum);

    // Thread 0 of each block writes result
    if (tid == 0) partial[blockIdx.x] = sum;
}

// ---- Kernel 2: reduce partial array (launch with 1 block) ----
__global__ void final_reduce_sum_kernel(
    const float* __restrict__ partial,
    float*       __restrict__ out,
    int num_blocks
) {
    int tid = threadIdx.x;
    float sum = 0.f;
    for (int i = tid; i < num_blocks; i += blockDim.x) sum += partial[i];
    sum = block_reduce_sum(sum);
    if (tid == 0) *out = sum;
}

// ---- Convenience: max reduce (same pattern) ----
__global__ void block_reduce_max_kernel(
    const float* __restrict__ in,
    float*       __restrict__ partial,
    int N
) {
    int tid = threadIdx.x;
    float mx = -FLT_MAX;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x)
        mx = fmaxf(mx, in[i]);
    mx = block_reduce_max(mx);
    if (tid == 0) partial[blockIdx.x] = mx;
}

// Usage pattern:
//   int threads = 256, blocks = min((N + threads - 1) / threads, 1024);
//   block_reduce_sum_kernel<<<blocks, threads>>>(in, partial, N);
//   final_reduce_sum_kernel<<<1, 256>>>(partial, out, blocks);


// ================================================================
//  5.  FlashAttention  (simplified, single-head, causal mask)
// ================================================================
//
//  Standard attention:   O = softmax(Q K^T / sqrt(d)) V
//
//  Problem: for long sequences, QK^T is an N×N matrix → O(N²) HBM.
//
//  FlashAttention insight:
//    Tile K and V into blocks of BLOCK_KV rows.
//    Process blocks left-to-right, maintaining online softmax state
//    per query row so we never materialize the full N×N matrix.
//
//  Online softmax state for query row i after seeing tiles 0..t:
//    m_i  = max over all seen scores          (running max)
//    l_i  = Σ exp(score - m_i), corrected     (running normalizer)
//    o_i  = Σ exp(score - m_i) * V_row        (running output)
//
//  Correction when new tile raises the max from m to m_new:
//    l_new  = l  * exp(m - m_new)  +  Σ_new exp(score - m_new)
//    o_new  = o  * exp(m - m_new)  +  Σ_new exp(score - m_new) * V
//
//  After all tiles: O[i] = o_i / l_i
//
//  This kernel:
//    • One block per BLOCK_Q consecutive query rows
//    • One thread per query row within the block (threadIdx.x = q offset)
//    • Assumes d ≤ 128 (head dimension fits in registers)
//    • BLOCK_KV must equal blockDim.x (= BLOCK_Q here for simplicity)

#define BLOCK_Q  32    // query rows per block
#define BLOCK_KV 32    // KV rows per tile  (= blockDim.x)
#define MAX_D    128   // max head dimension (register array bound)

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,  // [N, d]
    const float* __restrict__ K,  // [N, d]
    const float* __restrict__ V,  // [N, d]
    float*       __restrict__ O,  // [N, d]
    int N, int d,
    float scale                   // 1 / sqrt(d)
) {
    // Thread id = which query row within this block
    int local_q  = threadIdx.x;
    int q_row    = blockIdx.x * BLOCK_Q + local_q;  // global query index

    // ---- Shared memory tiles for K and V ----
    // Each tile holds BLOCK_KV rows × d cols.
    // We reload them once per KV tile (BLOCK_KV × d floats = 32×128×4 = 16 KB).
    __shared__ float Ks[BLOCK_KV][MAX_D];
    __shared__ float Vs[BLOCK_KV][MAX_D];

    // ---- Q row cached in registers (avoids repeated HBM reads) ----
    float q_reg[MAX_D] = {};
    if (q_row < N)
        for (int j = 0; j < d; ++j) q_reg[j] = Q[q_row * d + j];

    // ---- Online softmax accumulators (per query row, in registers) ----
    float m = -FLT_MAX;   // running max of all seen QK scores
    float l = 0.f;         // running normalizer (corrected sum of exp)
    float o[MAX_D] = {};   // running output accumulator

    // ---- Outer loop: stride over KV sequence in tiles ----
    for (int kv_start = 0; kv_start < N; kv_start += BLOCK_KV) {

        // ---- Collaborative load of K and V tiles ----
        // Thread `local_q` loads KV row `kv_start + local_q`.
        int kv_row = kv_start + local_q;
        if (kv_row < N) {
            for (int j = 0; j < d; ++j) {
                Ks[local_q][j] = K[kv_row * d + j];
                Vs[local_q][j] = V[kv_row * d + j];
            }
        } else {
            for (int j = 0; j < d; ++j) Ks[local_q][j] = Vs[local_q][j] = 0.f;
        }
        __syncthreads();

        // ---- Compute score tile: S[kv] = Q[q_row] · K[kv_start+kv] * scale ----
        int tile_len = min(BLOCK_KV, N - kv_start);
        float S[BLOCK_KV] = {};

        for (int kv = 0; kv < tile_len; ++kv) {
            float dot = 0.f;
            for (int j = 0; j < d; ++j) dot += q_reg[j] * Ks[kv][j];
            // Causal mask: future tokens get -inf score
            S[kv] = (q_row < N && kv_start + kv <= q_row) ? dot * scale : -FLT_MAX;
        }

        // ---- Online softmax update ----
        // Step 1: find tile max
        float m_tile = -FLT_MAX;
        for (int kv = 0; kv < tile_len; ++kv) m_tile = fmaxf(m_tile, S[kv]);

        // Step 2: correction factor for old running state
        float m_new      = fmaxf(m, m_tile);
        float correction = expf(m - m_new);   // rescale old l and o

        l *= correction;
        for (int j = 0; j < d; ++j) o[j] *= correction;

        // Step 3: incorporate new tile
        for (int kv = 0; kv < tile_len; ++kv) {
            float e = expf(S[kv] - m_new);    // exp of shifted score
            l += e;
            for (int j = 0; j < d; ++j) o[j] += e * Vs[kv][j];
        }
        m = m_new;

        __syncthreads();   // guard before next tile overwrites smem
    }

    // ---- Normalize and write output ----
    if (q_row < N)
        for (int j = 0; j < d; ++j)
            O[q_row * d + j] = o[j] / l;
}
// Launch:
//   dim3 grid(ceil(N / BLOCK_Q)), block(BLOCK_Q);
//   flash_attention_kernel<<<grid, block>>>(Q, K, V, O, N, d, 1.f/sqrtf(d));


// ================================================================
//  7.  Fused MHA  (naive, full QK^T materialized per query row)
// ================================================================
//
//  Differences from FlashAttention:
//    • Scores for the full context are written to shared memory first,
//      then softmax is applied in-place.
//    • This avoids the online-correction logic, trading simplicity for
//      O(N) shared memory per query row — only viable when N is small.
//    • Good for decode phase (num_q = 1, N = context length up to ~1024).
//
//  One block per query token.
//  Dynamic shared memory = (N + d) × sizeof(float).

__global__ void fused_mha_naive_kernel(
    const float* __restrict__ Q,   // [num_q, d]
    const float* __restrict__ K,   // [N, d]
    const float* __restrict__ V,   // [N, d]
    float*       __restrict__ O,   // [num_q, d]
    int num_q, int N, int d,
    float scale
) {
    int q_idx = blockIdx.x;    // which query token
    int tid   = threadIdx.x;

    // Dynamic smem layout:  [ scores[N] | v_acc[d] ]
    extern __shared__ float smem[];
    float* scores = smem;       // attention probabilities  [N]
    float* v_acc  = smem + N;   // output accumulator       [d]

    const float* q_ptr = Q + (long)q_idx * d;

    // ── Step 1: QK scores ──
    // Each thread computes scores for a stride of K rows.
    for (int kj = tid; kj < N; kj += blockDim.x) {
        float dot = 0.f;
        for (int h = 0; h < d; ++h)
            dot += q_ptr[h] * K[(long)kj * d + h];
        scores[kj] = dot * scale;
    }
    __syncthreads();

    // ── Step 2: Softmax over scores (in-place) ──

    // 2a: row max
    float local_max = -FLT_MAX;
    for (int j = tid; j < N; j += blockDim.x) local_max = fmaxf(local_max, scores[j]);
    float row_max = block_reduce_max(local_max);
    row_max = broadcast_from_zero(row_max);

    // 2b: exp(score - max) and partial sum
    float local_sum = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float e  = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    float row_sum = block_reduce_sum(local_sum);
    row_sum = broadcast_from_zero(row_sum);

    // 2c: normalize
    for (int j = tid; j < N; j += blockDim.x) scores[j] /= row_sum;
    __syncthreads();

    // ── Step 3: weighted sum over V  →  O = scores @ V ──
    // Initialize accumulator
    for (int h = tid; h < d; h += blockDim.x) v_acc[h] = 0.f;
    __syncthreads();

    // Each thread owns a slice of output dimensions [h].
    // For each h, sum over all context positions j.
    for (int h = tid; h < d; h += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < N; ++j)
            acc += scores[j] * V[(long)j * d + h];
        v_acc[h] = acc;
    }
    __syncthreads();

    // ── Write output ──
    for (int h = tid; h < d; h += blockDim.x)
        O[(long)q_idx * d + h] = v_acc[h];
}
// Launch:
//   size_t smem_bytes = (N + d) * sizeof(float);
//   fused_mha_naive_kernel<<<num_q, 256, smem_bytes>>>(Q, K, V, O, num_q, N, d, scale);


// ================================================================
//  Quick smoke-test main
// ================================================================
int main() {
    printf("Kernel definitions compiled successfully.\n");
    printf("Kernels included:\n");
    printf("  1. gemm_kernel\n");
    printf("  2. softmax_kernel\n");
    printf("  3. layernorm_kernel / rmsnorm_kernel\n");
    printf("  4. block_reduce_sum_kernel / block_reduce_max_kernel\n");
    printf("  5. flash_attention_kernel\n");
    printf("  7. fused_mha_naive_kernel\n");
    return 0;
}
