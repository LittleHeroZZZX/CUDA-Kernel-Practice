#pragma once

#include <cuda_runtime.h>
#include <float.h>

// Warp-level primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

// Cross-warp block reduce using shared memory.
__device__ __forceinline__ float block_reduce_sum(float val) {
  __shared__ float smem[32];
  int wid = threadIdx.x / 32;
  int lid = threadIdx.x % 32;
  int nw = (blockDim.x + 31) / 32;

  val = warp_reduce_sum(val);
  if (lid == 0)
    smem[wid] = val;
  __syncthreads();

  val = (wid == 0 && lid < nw) ? smem[lid] : 0.f;
  if (wid == 0)
    val = warp_reduce_sum(val);
  return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
  __shared__ float smem[32];
  int wid = threadIdx.x / 32;
  int lid = threadIdx.x % 32;
  int nw = (blockDim.x + 31) / 32;

  val = warp_reduce_max(val);
  if (lid == 0)
    smem[wid] = val;
  __syncthreads();

  val = (wid == 0 && lid < nw) ? smem[lid] : -FLT_MAX;
  if (wid == 0)
    val = warp_reduce_max(val);
  return val;
}

// Broadcast block result from thread 0 to all threads.
__device__ __forceinline__ float broadcast_from_zero(float val) {
  __shared__ float bcast;
  if (threadIdx.x == 0)
    bcast = val;
  __syncthreads();
  return bcast;
}
