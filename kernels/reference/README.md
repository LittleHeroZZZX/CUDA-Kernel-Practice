# LLM Kernels Split

This folder splits llm_kernels.cu into one CUDA file per operator.
Shared warp/block reduction helpers live in common.cuh.

## Files

- common.cuh
- gemm.cu
- softmax.cu
- layernorm.cu
- block_reduce.cu
- flash_attention.cu
- fused_mha.cu

## Build (example)

nvcc -O3 -arch=sm_80 gemm.cu -c
