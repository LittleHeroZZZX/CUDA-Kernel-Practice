import torch


def naive_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, causal=False)-> torch.Tensor:
    """
    Args:
        Q, K, V: (batch, n_heads, seq_len, head_dim)
        causal: 是否使用因果mask
    Returns:
        O: (batch, n_heads, seq_len, head_dim)
    """
    d = Q.shape[-1]
    seq_len = Q.shape[-2]
    
    # TODO 1: 计算 attention scores，注意 scale
    scores:torch.Tensor = Q @ K.mT * d**-0.5
    
    # TODO 2: 如果 causal=True，把未来位置 mask 成 -inf
    if causal:
        scores += (torch.ones(seq_len, seq_len, dtype=scores.dtype, device=scores.device) * -torch.inf).triu(1)
        

    
    # TODO 3: softmax
    scores  = scores - scores.max(dim=-1, keepdim=True)[0]
    exp = scores.exp()
    attn = exp / exp.sum(dim=-1, keepdim=True)
    
    # TODO 4: 乘以 V
    O = attn @ V
    
    return O


if __name__ == "__main__":
    B, H, N, d = 2, 4, 1024, 64
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    
    out = naive_attention(Q, K, V, causal=True)
    print("Output shape:", out.shape)
    print("No NaN:", not torch.isnan(out).any().item())

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("Max diff vs PyTorch:", (out - ref).abs().max().item())