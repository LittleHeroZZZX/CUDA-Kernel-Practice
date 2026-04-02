"""Triton Fused MHA — mirrors fused_mha.cu (naive: materialize all scores).

CUDA structure (one block per query):
  step 1 — stride loop: scores[kj] = dot(q, K[kj]) * scale    (materialize N scores)
  step 2 — block_reduce_max → broadcast → exp(scores - max)
          → block_reduce_sum → normalize (softmax over N scores in smem)
  step 3 — stride loop over d: acc += scores[j] * V[j * d + h]  (weighted sum)

Triton mapping: one program per query; smem arrays become register tensors
  (BLOCK_N covers N keys, BLOCK_D covers the head dimension).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    num_q, N, d, scale,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per query — mirrors blockIdx.x = q_idx.
    q_idx  = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    kv_mask = offs_n < N
    d_mask  = offs_d < d

    # Load query vector — mirrors q_ptr = Q + q_idx * d.
    q = tl.load(Q_ptr + q_idx * stride_qn + offs_d * stride_qd, mask=d_mask, other=0.0)

    # Load all K [BLOCK_N, BLOCK_D] — mirrors iterating over kj with inner h loop.
    K = tl.load(
        K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Step 1: scores[kj] = dot(q, K[kj]) * scale — mirrors the kj stride loop.
    scores = tl.sum(K * q[None, :], axis=1) * scale   # [BLOCK_N]
    scores = tl.where(kv_mask, scores, float('-inf'))

    # Step 2: softmax — mirrors block_reduce_max + exp + block_reduce_sum.
    row_max = tl.max(scores, axis=0)
    e = tl.exp(tl.where(kv_mask, scores - row_max, float('-inf')))
    row_sum = tl.sum(tl.where(kv_mask, e, 0.0), axis=0)
    attn = tl.where(kv_mask, e / row_sum, 0.0)        # [BLOCK_N]

    # Load all V [BLOCK_N, BLOCK_D] — mirrors the v_acc accumulation loop.
    V = tl.load(
        V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
        mask=kv_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    # Step 3: out[h] = sum_j attn[j] * V[j, h] — mirrors the h stride loop.
    out = tl.sum(attn[:, None] * V, axis=0)            # [BLOCK_D]
    tl.store(O_ptr + q_idx * stride_on + offs_d * stride_od, out, mask=d_mask)


def forward(q, k, v):
    num_q, d = q.shape
    N = k.shape[0]
    o = torch.empty_like(q)
    scale  = 1.0 / (d ** 0.5)
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_D = triton.next_power_of_2(d)
    _kernel[(num_q,)](
        q, k, v, o,
        num_q, N, d, scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o
