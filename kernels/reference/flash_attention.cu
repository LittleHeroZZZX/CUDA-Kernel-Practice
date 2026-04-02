#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include "common.cuh"

#define BLOCK_Q 32
#define BLOCK_KV 32
#define MAX_D 128

__global__ void flash_attention_kernel(const float *__restrict__ Q,
                                       const float *__restrict__ K,
                                       const float *__restrict__ V,
                                       float *__restrict__ O, int N, int d,
                                       float scale) {
  int local_q = threadIdx.x;
  int q_row = blockIdx.x * BLOCK_Q + local_q;

  __shared__ float Ks[BLOCK_KV][MAX_D];
  __shared__ float Vs[BLOCK_KV][MAX_D];

  float q_reg[MAX_D] = {};
  if (q_row < N) {
    for (int j = 0; j < d; ++j)
      q_reg[j] = Q[q_row * d + j];
  }

  float m = -FLT_MAX;
  float l = 0.f;
  float o[MAX_D] = {};

  for (int kv_start = 0; kv_start < N; kv_start += BLOCK_KV) {
    int kv_row = kv_start + local_q;
    if (kv_row < N) {
      for (int j = 0; j < d; ++j) {
        Ks[local_q][j] = K[kv_row * d + j];
        Vs[local_q][j] = V[kv_row * d + j];
      }
    } else {
      for (int j = 0; j < d; ++j) {
        Ks[local_q][j] = 0.f;
        Vs[local_q][j] = 0.f;
      }
    }
    __syncthreads();

    int tile_len = min(BLOCK_KV, N - kv_start);
    float S[BLOCK_KV] = {};
    for (int kv = 0; kv < tile_len; ++kv) {
      float dot = 0.f;
      for (int j = 0; j < d; ++j)
        dot += q_reg[j] * Ks[kv][j];
      S[kv] = (q_row < N && kv_start + kv <= q_row) ? dot * scale : -FLT_MAX;
    }

    float m_tile = -FLT_MAX;
    for (int kv = 0; kv < tile_len; ++kv)
      m_tile = fmaxf(m_tile, S[kv]);

    float m_new = fmaxf(m, m_tile);
    float correction = expf(m - m_new);

    l *= correction;
    for (int j = 0; j < d; ++j)
      o[j] *= correction;

    for (int kv = 0; kv < tile_len; ++kv) {
      float e = expf(S[kv] - m_new);
      l += e;
      for (int j = 0; j < d; ++j)
        o[j] += e * Vs[kv][j];
    }
    m = m_new;

    __syncthreads();
  }

  if (q_row < N) {
    for (int j = 0; j < d; ++j)
      O[q_row * d + j] = o[j] / l;
  }
}
