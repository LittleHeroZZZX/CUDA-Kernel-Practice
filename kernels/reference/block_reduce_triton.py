"""Triton block reduce — mirrors block_reduce.cu (two-kernel sum reduction).

CUDA structure:
  kernel 1 (block_reduce_sum_kernel):  grid-stride loop, each block reduces its
            chunk to partial[blockIdx.x] via block_reduce_sum().
  kernel 2 (final_reduce_sum_kernel):  single block reduces all partials.

Triton mapping: exactly two kernels with the same two-phase split.
"""
import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024  # threads per block — mirrors typical CUDA launch config


@triton.jit
def _partial_sum_kernel(X_ptr, Partial_ptr, N, BLOCK: tl.constexpr):
    # One program per chunk — mirrors block_reduce_sum_kernel.
    # Each program does a grid-stride reduction over BLOCK elements,
    # then stores the block sum to partial[bid].
    bid = tl.program_id(0)
    offs = bid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X_ptr + offs, mask=offs < N, other=0.0)
    # tl.sum mirrors warp_reduce + smem cross-warp reduce in block_reduce_sum().
    partial = tl.sum(x, axis=0)
    tl.store(Partial_ptr + bid, partial)


@triton.jit
def _final_sum_kernel(Partial_ptr, Out_ptr, num_blocks, BLOCK: tl.constexpr):
    # Single program reduces all partial sums — mirrors final_reduce_sum_kernel.
    offs = tl.arange(0, BLOCK)
    partial = tl.load(Partial_ptr + offs, mask=offs < num_blocks, other=0.0)
    total = tl.sum(partial, axis=0)
    tl.store(Out_ptr, total)


def forward(x):
    N = x.numel()
    x_flat = x.reshape(-1)
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    partial = torch.empty(num_blocks, device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.float32)

    _partial_sum_kernel[(num_blocks,)](x_flat, partial, N, BLOCK=BLOCK_SIZE)

    # Final kernel needs BLOCK >= num_blocks (power of 2 for tl.arange).
    BLOCK2 = triton.next_power_of_2(num_blocks)
    _final_sum_kernel[(1,)](partial, out, num_blocks, BLOCK=BLOCK2)

    return out
