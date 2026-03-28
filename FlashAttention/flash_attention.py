import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d: tl.constexpr,
    scale,
    BLOCK_N: tl.constexpr,  # Br: Q的块大小
    BLOCK_M: tl.constexpr,  # Bc: KV的块大小
    causal: tl.constexpr,
):
    # 每个 program 负责一个 (batch, head, Q块)
    pid = tl.program_id(0)   # Q块的索引
    bid = tl.program_id(1)   # batch index
    hid = tl.program_id(2)   # head index

    # 当前 Q 块的起始行
    q_start = pid * BLOCK_N

    # 各个指针偏移到正确的 batch 和 head
    Q_ptr = Q_ptr + bid * stride_qb + hid * stride_qh
    K_ptr = K_ptr + bid * stride_kb + hid * stride_kh
    V_ptr = V_ptr + bid * stride_vb + hid * stride_vh
    O_ptr = O_ptr + bid * stride_ob + hid * stride_oh

    # 加载 Q 块: shape (BLOCK_N, d)
    # TODO 1: 用 tl.arange 构造行列 offsets，用 tl.load 加载，注意 boundary mask
    q_row_offsets = q_start + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    d_offsets = tl.arange(0, d)                  # (d,) 注意 d 必须是2的幂
    q_mask = q_row_offsets[:, None] < N                # (BLOCK_N, 1)  防止越界

    Q_block = tl.load(
        Q_ptr + q_row_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd,
        mask=q_mask,
        other=0.0
    )  # (BLOCK_N, d)

    # 初始化 online softmax 的状态
    # TODO 2: m 初始化为 -inf, l 初始化为 0, O 初始化为 0
    m = tl.full((BLOCK_N,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_N,), dtype=tl.float32) 
    O_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)

    # 内层循环：遍历所有 KV 块
    # TODO 3: causal 时只需要遍历到当前 Q 块，填写 kv_end
    kv_end = tl.cdiv(N, BLOCK_M)  # 非causal：遍历全部
    if causal:
        kv_end = tl.cdiv(q_start + BLOCK_N, BLOCK_M)  # causal：只看左边

    for j in range(0, kv_end):
        kv_start = j * BLOCK_M
        kv_row_offsets = kv_start + tl.arange(0, BLOCK_M)
        kv_mask = kv_row_offsets[:, None] < N

        # 加载 K 块和 V 块
        K_block = tl.load(
            K_ptr + kv_row_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd,
            mask=kv_mask,
            other=0.0
        )  # (BLOCK_M, d)

        V_block = tl.load(
            V_ptr + kv_row_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd,
            mask=kv_mask,
            other=0.0
        )  # (BLOCK_M, d)

        # TODO 4: 计算 S = Q @ K^T * scale, shape (BLOCK_N, BLOCK_M)
        S = tl.dot(Q_block, tl.trans(K_block)) * scale

        # TODO 5: causal mask：当前 KV 块中超出 q_row 的位置设为 -inf
        if causal:
            causal_mask = q_row_offsets[:, None] >= kv_row_offsets[None, :]
            S = tl.where(causal_mask, S, float('-inf'))

        # TODO 6: online softmax 更新
        # 6a. 计算这个块的行最大值
        m_new = tl.maximum(m, tl.max(S, axis=1))  # (BLOCK_N,)

        # 6b. 用新最大值修正 O_acc 和 l
        alpha = tl.exp(m - m_new)   # 旧最大值相对新最大值的修正系数
        O_acc = O_acc * alpha[:, None]
        l = l * alpha

        # 6c. 计算当前块的 exp(S - m_new)
        P = tl.exp(S - m_new[:, None])  # (BLOCK_N, BLOCK_M)

        # 6d. 更新 l 和 O_acc
        l = l + tl.sum(P, axis=1)
        O_acc = O_acc + tl.dot(P, V_block.to(tl.float32))

        # 更新 m
        m = m_new

    # TODO 7: 最终归一化
    O_acc = O_acc / l[:, None]

    # TODO 8: 写回结果到 HBM
    o_mask = q_row_offsets[:, None] < N
    tl.store(
        O_ptr + q_row_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od,
        O_acc.to(Q_block.dtype),  # 输出类型和输入Q相同
        mask=o_mask
    )


def flash_attention(Q, K, V, causal=False):
    B, H, N, d = Q.shape
    assert d in (16, 32, 64, 128), "d 必须是2的幂"

    O = torch.zeros_like(Q)
    scale = 1.0 / math.sqrt(d)

    # 按论文公式计算 block size
    device = torch.cuda.current_device()
    shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
    element_size = Q.element_size()  # bf16=2, fp32=4
    
    Bc = min(triton.next_power_of_2(shared_mem // (4 * d * element_size)), 64)
    Br = min(Bc, d)

    grid = (triton.cdiv(N, Br), B, H)

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, d, scale,
        BLOCK_N=Br,
        BLOCK_M=Bc,
        causal=causal,
    )
    return O

if __name__ == "__main__":
    B, H, N, d = 2, 4, 1024, 64
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

    out = flash_attention(Q, K, V, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    print("Max diff:", (out - ref).abs().max().item())
    print("Mean diff:", (out - ref).abs().mean().item())