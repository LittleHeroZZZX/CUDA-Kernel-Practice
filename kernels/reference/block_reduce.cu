#include <cuda_runtime.h>
#include <float.h>

#include "common.cuh"

__global__ void block_reduce_sum_kernel(const float *__restrict__ in,
                                        float *__restrict__ partial, int N) {
  int tid = threadIdx.x;

  float sum = 0.f;
  for (int i = blockIdx.x * blockDim.x + tid; i < N;
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }

  sum = block_reduce_sum(sum);
  if (tid == 0)
    partial[blockIdx.x] = sum;
}

__global__ void final_reduce_sum_kernel(const float *__restrict__ partial,
                                        float *__restrict__ out,
                                        int num_blocks) {
  int tid = threadIdx.x;
  float sum = 0.f;
  for (int i = tid; i < num_blocks; i += blockDim.x)
    sum += partial[i];
  sum = block_reduce_sum(sum);
  if (tid == 0)
    *out = sum;
}

__global__ void block_reduce_max_kernel(const float *__restrict__ in,
                                        float *__restrict__ partial, int N) {
  int tid = threadIdx.x;
  float mx = -FLT_MAX;
  for (int i = blockIdx.x * blockDim.x + tid; i < N;
       i += blockDim.x * gridDim.x) {
    mx = fmaxf(mx, in[i]);
  }
  mx = block_reduce_max(mx);
  if (tid == 0)
    partial[blockIdx.x] = mx;
}
