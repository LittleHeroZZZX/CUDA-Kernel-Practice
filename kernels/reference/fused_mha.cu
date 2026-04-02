#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#include "common.cuh"

__global__ void fused_mha_naive_kernel(const float *__restrict__ Q,
                                       const float *__restrict__ K,
                                       const float *__restrict__ V,
                                       float *__restrict__ O, int num_q, int N,
                                       int d, float scale) {
  int q_idx = blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float smem[];
  float *scores = smem;
  float *v_acc = smem + N;

  const float *q_ptr = Q + (long)q_idx * d;

  for (int kj = tid; kj < N; kj += blockDim.x) {
    float dot = 0.f;
    for (int h = 0; h < d; ++h) {
      dot += q_ptr[h] * K[(long)kj * d + h];
    }
    scores[kj] = dot * scale;
  }
  __syncthreads();

  float local_max = -FLT_MAX;
  for (int j = tid; j < N; j += blockDim.x)
    local_max = fmaxf(local_max, scores[j]);
  float row_max = block_reduce_max(local_max);
  row_max = broadcast_from_zero(row_max);

  float local_sum = 0.f;
  for (int j = tid; j < N; j += blockDim.x) {
    float e = expf(scores[j] - row_max);
    scores[j] = e;
    local_sum += e;
  }
  float row_sum = block_reduce_sum(local_sum);
  row_sum = broadcast_from_zero(row_sum);

  for (int j = tid; j < N; j += blockDim.x)
    scores[j] /= row_sum;
  __syncthreads();

  for (int h = tid; h < d; h += blockDim.x)
    v_acc[h] = 0.f;
  __syncthreads();

  for (int h = tid; h < d; h += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < N; ++j)
      acc += scores[j] * V[(long)j * d + h];
    v_acc[h] = acc;
  }
  __syncthreads();

  for (int h = tid; h < d; h += blockDim.x) {
    O[(long)q_idx * d + h] = v_acc[h];
  }
}
