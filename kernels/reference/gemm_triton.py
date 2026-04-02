"""Triton GEMM — mirrors gemm.cu (tiled shared-memory matmul).

CUDA constants:  TILE_M=64  TILE_N=64  TILE_K=16  WM=4  WN=4
Triton mapping:  BLOCK_M=64 BLOCK_N=64 BLOCK_K=16  (tl.dot replaces the WM×WN register loop)
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16


@triton.jit
def _kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per (BLOCK_M, BLOCK_N) output tile.
    # Mirrors: blockIdx.y * TILE_M row-group, blockIdx.x * TILE_N col-group.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-tile loop — mirrors the k0 loop stepping by TILE_K.
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K] — mirrors loading As[TILE_M][TILE_K].
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        # Load B tile [BLOCK_K, BLOCK_N] — mirrors loading Bs[TILE_K][TILE_N].
        b = tl.load(
            B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        # tl.dot replaces the unrolled kk / WM×WN register accumulation.
        # allow_tf32=False: match CUDA's full fp32 precision (no TF32 rounding).
        acc += tl.dot(a, b, allow_tf32=False)

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def forward(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
