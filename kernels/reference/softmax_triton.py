"""Triton softmax — mirrors softmax.cu (3-pass per-row softmax).

CUDA structure (one block per row):
  pass 1 — stride loop → block_reduce_max → broadcast → row_max
  pass 2 — stride loop: exp(x - row_max) → store exp, accumulate → block_reduce_sum
  pass 3 — stride loop: divide by row_sum

Triton mapping: one program per row; BLOCK covers the whole row so the three
passes collapse to three vector ops, but the ordering mirrors the CUDA passes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(X_ptr, Out_ptr, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols

    # Pass 1: load row, find max — mirrors block_reduce_max + broadcast.
    x = tl.load(X_ptr + row * cols + offs, mask=mask, other=float('-inf'))
    row_max = tl.max(x, axis=0)

    # Pass 2: exp(x - max) and sum — mirrors exp store + block_reduce_sum.
    e = tl.exp(tl.where(mask, x - row_max, float('-inf')))
    row_sum = tl.sum(tl.where(mask, e, 0.0), axis=0)

    # Pass 3: normalize — mirrors the final divide loop.
    out = tl.where(mask, e / row_sum, 0.0)
    tl.store(Out_ptr + row * cols + offs, out, mask=mask)


def forward(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    _kernel[(rows,)](x, out, cols, BLOCK=BLOCK)
    return out
