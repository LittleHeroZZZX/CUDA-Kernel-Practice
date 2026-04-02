"""Triton Flash Attention — mirrors flash_attention.cu (online softmax, causal).

CUDA structure (BLOCK_Q=32, BLOCK_KV=32):
  - One CUDA block per BLOCK_Q tile of queries; threadIdx.x = query within tile.
  - Each thread keeps q_reg[d], m, l, o[d] in registers.
  - Scans KV tiles: loads Ks/Vs into shared mem, computes dot products, updates
    running (m, l, O) with the online-softmax correction factor.
  - Causal mask: S[kv] = -inf if kv_start + kv > q_row.

Triton mapping:
  - One program per BLOCK_Q tile; tl.dot replaces the per-thread dot-product loop.
  - Load K transposed [BLOCK_D, BLOCK_KV] to use Q @ K_T for the score matrix.
  - Online softmax state (m, l, O) lives in registers across loop iterations.
"""
import torch
import triton
import triton.language as tl

BLOCK_Q  = 32
BLOCK_KV = 32


@triton.jit
def _kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    N, d, scale,
    stride_n, stride_d,
    BLOCK_Q:  tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D:  tl.constexpr,
):
    q_start = tl.program_id(0) * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile [BLOCK_Q, BLOCK_D] — mirrors loading q_reg from global Q.
    Q = tl.load(
        Q_ptr + offs_q[:, None] * stride_n + offs_d[None, :] * stride_d,
        mask=(offs_q[:, None] < N) & (offs_d[None, :] < d),
        other=0.0,
    )

    # Online-softmax state — mirrors float m=-FLT_MAX, l=0, o[MAX_D]=0.
    m = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    # Scan KV tiles — mirrors for (kv_start = 0; kv_start < N; kv_start += BLOCK_KV).
    for kv_start in range(0, N, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)

        # Load K transposed [BLOCK_D, BLOCK_KV] for Q @ K_T dot product.
        # Mirrors: loading Ks[BLOCK_KV][MAX_D] and computing dot(q_reg, Ks[kv]).
        K_T = tl.load(
            K_ptr + offs_d[:, None] * stride_d + offs_kv[None, :] * stride_n,
            mask=(offs_d[:, None] < d) & (offs_kv[None, :] < N),
            other=0.0,
        )
        # Load V tile [BLOCK_KV, BLOCK_D] — mirrors loading Vs[BLOCK_KV][MAX_D].
        V = tl.load(
            V_ptr + offs_kv[:, None] * stride_n + offs_d[None, :] * stride_d,
            mask=(offs_kv[:, None] < N) & (offs_d[None, :] < d),
            other=0.0,
        )

        # Score matrix [BLOCK_Q, BLOCK_KV] — mirrors dot(q_reg[j], Ks[kv][j]) * scale.
        S = tl.dot(Q, K_T, allow_tf32=False) * scale

        # Causal mask + bounds — mirrors: (q_row >= kv_start + kv) ? score : -inf.
        causal   = offs_q[:, None] >= offs_kv[None, :]
        in_bound = offs_kv[None, :] < N
        S = tl.where(causal & in_bound, S, float('-inf'))

        # Online-softmax update — mirrors the m_new / correction / l / o block.
        m_tile = tl.max(S, axis=1)                    # [BLOCK_Q]
        m_new  = tl.maximum(m, m_tile)
        alpha  = tl.exp(m - m_new)                    # correction for accumulated state
        P      = tl.exp(S - m_new[:, None])           # [BLOCK_Q, BLOCK_KV]

        l = l * alpha + tl.sum(P, axis=1)
        O = O * alpha[:, None] + tl.dot(P, V, allow_tf32=False)
        m = m_new

    # Normalize — mirrors O[q_row * d + j] = o[j] / l.
    O = O / l[:, None]
    tl.store(
        O_ptr + offs_q[:, None] * stride_n + offs_d[None, :] * stride_d,
        O,
        mask=(offs_q[:, None] < N) & (offs_d[None, :] < d),
    )


def forward(q, k, v):
    N, d = q.shape
    o = torch.empty_like(q)
    scale = 1.0 / (d ** 0.5)
    BLOCK_D = triton.next_power_of_2(d)
    grid = (triton.cdiv(N, BLOCK_Q),)
    _kernel[grid](
        q, k, v, o,
        N, d, scale,
        q.stride(0), q.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
    )
    return o
