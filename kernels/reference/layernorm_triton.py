"""Triton LayerNorm/RMSNorm — mirrors layernorm.cu.

CUDA layernorm_kernel (one block per row):
  pass 1 — block_reduce_sum / cols  → mean
  pass 2 — block_reduce_sum(d²) / cols → var → inv_std
  pass 3 — (x - mean) * inv_std * gamma + beta

CUDA rmsnorm_kernel (one block per row):
  pass 1 — block_reduce_sum(x²) / cols → rms → inv_rms
  pass 2 — x * inv_rms * gamma

Triton mapping: one program per row; two separate kernels mirror the two CUDA kernels.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    X_ptr, Gamma_ptr, Beta_ptr, Out_ptr,
    cols, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols

    x = tl.load(X_ptr + row * cols + offs, mask=mask, other=0.0)
    gamma = tl.load(Gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(Beta_ptr + offs, mask=mask, other=0.0)

    # Pass 1: mean — mirrors block_reduce_sum(local_sum) / cols + broadcast.
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / cols

    # Pass 2: variance — mirrors block_reduce_sum(d*d) / cols + rsqrtf.
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / cols
    inv_std = tl.rsqrt(var + eps)

    # Pass 3: normalize + affine — mirrors ro[j] = (rx[j]-mean)*inv_std*gamma+beta.
    out = d * inv_std * gamma + tl.where(mask, beta, 0.0)
    tl.store(Out_ptr + row * cols + offs, out, mask=mask)


@triton.jit
def _rmsnorm_kernel(
    X_ptr, Gamma_ptr, Out_ptr,
    cols, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols

    x = tl.load(X_ptr + row * cols + offs, mask=mask, other=0.0)
    gamma = tl.load(Gamma_ptr + offs, mask=mask, other=1.0)

    # Pass 1: mean of squares — mirrors block_reduce_sum(local_ss) / cols.
    ss = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / cols
    inv_rms = tl.rsqrt(ss + eps)

    # Pass 2: normalize — mirrors ro[j] = rx[j] * inv_rms * gamma[j].
    out = tl.where(mask, x * inv_rms * gamma, 0.0)
    tl.store(Out_ptr + row * cols + offs, out, mask=mask)


def forward(x, gamma, beta=None, variant="layernorm", eps=1e-5):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)

    if variant == "rmsnorm":
        _rmsnorm_kernel[(rows,)](x, gamma, out, cols, eps, BLOCK=BLOCK)
    else:
        if beta is None:
            raise ValueError("beta required for layernorm")
        _layernorm_kernel[(rows,)](x, gamma, beta, out, cols, eps, BLOCK=BLOCK)
    return out
