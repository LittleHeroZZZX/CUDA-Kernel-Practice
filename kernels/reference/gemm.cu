#include <cuda_runtime.h>

#include "common.cuh"

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WM 4
#define WN 4

__global__ void gemm_kernel(const float *__restrict__ A,
                            const float *__restrict__ B, float *__restrict__ C,
                            int M, int N, int K) {
  __shared__ float As[TILE_M][TILE_K];
  __shared__ float Bs[TILE_K][TILE_N];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;
  int nt = blockDim.x * blockDim.y;

  int row0 = blockIdx.y * TILE_M + ty * WM;
  int col0 = blockIdx.x * TILE_N + tx * WN;

  float acc[WM][WN] = {};

  for (int k0 = 0; k0 < K; k0 += TILE_K) {
    for (int i = tid; i < TILE_M * TILE_K; i += nt) {
      int sm_r = i / TILE_K;
      int sm_c = i % TILE_K;
      int gr = blockIdx.y * TILE_M + sm_r;
      int gc = k0 + sm_c;
      As[sm_r][sm_c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
    }

    for (int i = tid; i < TILE_K * TILE_N; i += nt) {
      int sm_r = i / TILE_N;
      int sm_c = i % TILE_N;
      int gr = k0 + sm_r;
      int gc = blockIdx.x * TILE_N + sm_c;
      Bs[sm_r][sm_c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE_K; ++kk) {
      float a[WM];
      float b[WN];
#pragma unroll
      for (int m = 0; m < WM; ++m)
        a[m] = As[ty * WM + m][kk];
#pragma unroll
      for (int n = 0; n < WN; ++n)
        b[n] = Bs[kk][tx * WN + n];
#pragma unroll
      for (int m = 0; m < WM; ++m) {
#pragma unroll
        for (int n = 0; n < WN; ++n) {
          acc[m][n] += a[m] * b[n];
        }
      }
    }

    __syncthreads();
  }

#pragma unroll
  for (int m = 0; m < WM; ++m) {
#pragma unroll
    for (int n = 0; n < WN; ++n) {
      int r = row0 + m;
      int c = col0 + n;
      if (r < M && c < N)
        C[r * N + c] = acc[m][n];
    }
  }
}
