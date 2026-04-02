#include <cuda_runtime.h>
#include <math.h>

#include "common.cuh"

__global__ void layernorm_kernel(const float *__restrict__ x,
                                 const float *__restrict__ gamma,
                                 const float *__restrict__ beta,
                                 float *__restrict__ out, int cols, float eps) {
  int row = blockIdx.x;
  const float *rx = x + (long)row * cols;
  float *ro = out + (long)row * cols;
  int tid = threadIdx.x;

  float local_sum = 0.f;
  for (int j = tid; j < cols; j += blockDim.x) {
    local_sum += rx[j];
  }
  float mean = block_reduce_sum(local_sum) / cols;
  mean = broadcast_from_zero(mean);

  float local_var = 0.f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float d = rx[j] - mean;
    local_var += d * d;
  }
  float var = block_reduce_sum(local_var) / cols;
  float inv_std = rsqrtf(broadcast_from_zero(var) + eps);

  for (int j = tid; j < cols; j += blockDim.x) {
    ro[j] = (rx[j] - mean) * inv_std * gamma[j] + beta[j];
  }
}

__global__ void rmsnorm_kernel(const float *__restrict__ x,
                               const float *__restrict__ gamma,
                               float *__restrict__ out, int cols, float eps) {
  int row = blockIdx.x;
  const float *rx = x + (long)row * cols;
  float *ro = out + (long)row * cols;
  int tid = threadIdx.x;

  float local_ss = 0.f;
  for (int j = tid; j < cols; j += blockDim.x) {
    local_ss += rx[j] * rx[j];
  }

  float rms = block_reduce_sum(local_ss) / cols;
  float inv_rms = rsqrtf(broadcast_from_zero(rms) + eps);

  for (int j = tid; j < cols; j += blockDim.x) {
    ro[j] = rx[j] * inv_rms * gamma[j];
  }
}
