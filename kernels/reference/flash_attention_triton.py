"""Triton Flash Attention — 4D tensors [B, H, N, d], causal mask.

CUDA structure (BLOCK_Q=32, BLOCK_KV=32):
  - Grid x = ceil(N / BLOCK_Q), grid y = B * H.
  - blockIdx.y selects the (batch, head) slice; within that slice the
    algorithm is identical to the original single-head kernel.
  - Each thread keeps q_reg[d], m, l, o[d] in registers.
  - Scans KV tiles: loads Ks/Vs into shared mem, computes dot products,
    updates running (m, l, O) with the online-softmax correction factor.
  - Causal mask: S[kv] = -inf if kv_start + kv > q_row.

Triton mapping:
  - program_id(0) = which BLOCK_Q tile of queries (like blockIdx.x).
  - program_id(1) = which (batch, head) pair    (like blockIdx.y).
  - bh_off = pid_bh * stride_bh shifts the base pointer to the right slice.
  - tl.dot replaces the per-thread dot-product loop.
"""
import torch
import triton
import triton.language as tl

BLOCK_Q  = 32
BLOCK_KV = 32


@triton.jit
def _kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    N,
    d,
    scale,
    stride_bh,
    stride_n,
    stride_d,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_q = tl.program_id(0)  # which BLOCK_Q tile of queries
    pid_bh = tl.program_id(1)  # which (batch, head) pair

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    # Base offset for this (batch, head) slice.
    # For contiguous [B, H, N, d]: stride_bh = N*d, so bh_off = pid_bh * N * d.
    bh_off = pid_bh * stride_bh

    # Load Q tile [BLOCK_Q, BLOCK_D].
    Q = tl.load(
        Q_ptr + bh_off + offs_q[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=(offs_q[:, None] < N) & (offs_d[None, :] < d),
        other=0.0,
    )

    # Online-softmax state.
    m = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # Scan KV tiles.
    for kv_start in range(0, N, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)

        # Load K transposed [BLOCK_D, BLOCK_KV] for Q @ K_T.
        K_T = tl.load(
            K_ptr + bh_off + offs_d[:, None] * stride_d + offs_kv[None, :] * stride_n,
            mask=(offs_d[:, None] < d) & (offs_kv[None, :] < N),
            other=0.0,
        )
        # Load V tile [BLOCK_KV, BLOCK_D].
        V = tl.load(
            V_ptr + bh_off + offs_kv[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=(offs_kv[:, None] < N) & (offs_d[None, :] < d),
            other=0.0,
        )

        # Score matrix [BLOCK_Q, BLOCK_KV].
        S = tl.dot(Q, K_T, allow_tf32=False) * scale

        # Causal mask + bounds.
        causal   = offs_q[:, None] >= offs_kv[None, :]
        in_bound = offs_kv[None, :] < N
        S = tl.where(causal & in_bound, S, float('-inf'))

        # Online-softmax update.
        m_tile = tl.max(S, axis=1)
        m_new  = tl.maximum(m, m_tile)
        alpha = tl.exp(m - m_new)
        P = tl.exp(S - m_new[:, None])

        l = l * alpha + tl.sum(P, axis=1)
        O = O * alpha[:, None] + tl.dot(P, V, allow_tf32=False)
        m = m_new

    # Normalize and write output.
    O = O / l[:, None]
    tl.store(
        O_ptr + bh_off + offs_q[:, None] * stride_n + offs_d[None, :] * stride_d,
        O,
        mask=(offs_q[:, None] < N) & (offs_d[None, :] < d),
    )


def forward(q, k, v):
    B, H, N, d = q.shape
    o = torch.empty_like(q)
    scale = 1.0 / (d ** 0.5)
    BLOCK_D = triton.next_power_of_2(d)
    # grid x: query tiles; grid y: (batch, head) pairs.
    grid = (triton.cdiv(N, BLOCK_Q), B * H)
    # stride_bh: bytes between consecutive (batch, head) slices = q.stride(1).
    # For contiguous [B, H, N, d] this equals N*d.
    _kernel[grid](
        q,
        k,
        v,
        o,
        N,
        d,
        scale,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        BLOCK_D=BLOCK_D,
    )
    return o
