#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "common.cuh"

__global__ void softmax_kernel(const float *__restrict__ x,
                               float *__restrict__ out, int cols) {
  int row = blockIdx.x;
  const float *rx = x + (long)row * cols;
  float *ro = out + (long)row * cols;
  int tid = threadIdx.x;

  float local_max = -FLT_MAX;
  for (int j = tid; j < cols; j += blockDim.x) {
    local_max = fmaxf(local_max, rx[j]);
  }

  float row_max = block_reduce_max(local_max);
  row_max = broadcast_from_zero(row_max);

  float local_sum = 0.f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float e = expf(rx[j] - row_max);
    ro[j] = e;
    local_sum += e;
  }

  float row_sum = block_reduce_sum(local_sum);
  row_sum = broadcast_from_zero(row_sum);

  for (int j = tid; j < cols; j += blockDim.x) {
    ro[j] /= row_sum;
  }
}
