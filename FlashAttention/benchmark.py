import torch
from flash_attn import flash_attn_func

from flash_attention import flash_attention
from naive_attention import naive_attention


def benchmark(fn, *args, warmup=10, rep=100, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / rep

def get_hbm_bytes(B, H, N, d):
    naive_bytes = (
        3 * B * H * N * d * 4 +
        2 * B * H * N * N * 4 +
        2 * B * H * N * N * 4 +
        B * H * N * d * 4
    )
    flash_bytes = 4 * B * H * N * d * 4
    return naive_bytes, flash_bytes

if __name__ == "__main__":
    configs = [
        (1, 4,  512,  64),
        (1, 4, 1024,  64),
        (1, 4, 2048,  64),
        (2, 4, 2048,  64),
        (2, 8, 2048,  64),
        (2, 8, 4096,  64),
    ]

    print(f"{'Config':<28} {'Naive(ms)':>9} {'Ours(ms)':>9} {'FA2(ms)':>9} {'vs Naive':>9} {'vs FA2':>9}")
    print("-" * 80)

    for (B, H, N, d) in configs:
        # naive 和 ours 用 float16
        Q32 = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
        K32 = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)
        V32 = torch.randn(B, H, N, d, device='cuda', dtype=torch.float16)

        # flash-attn 官方只支持 float16/bfloat16
        # 官方接口: (B, N, H, d)，注意维度顺序不同
        Q16 = torch.randn(B, N, H, d, device='cuda', dtype=torch.float16)
        K16 = torch.randn(B, N, H, d, device='cuda', dtype=torch.float16)
        V16 = torch.randn(B, N, H, d, device='cuda', dtype=torch.float16)

        t_naive = benchmark(naive_attention, Q32, K32, V32, causal=True)
        t_ours  = benchmark(flash_attention, Q32, K32, V32, causal=True)
        t_fa2   = benchmark(flash_attn_func, Q16, K16, V16, causal=True)

        speedup_vs_naive = t_naive / t_ours
        ratio_vs_fa2     = t_ours / t_fa2   # 我们比 FA2 慢多少倍

        config_str = f"B={B} H={H} N={N} d={d}"
        print(f"{config_str:<28} {t_naive:>9.3f} {t_ours:>9.3f} {t_fa2:>9.3f} {speedup_vs_naive:>8.2f}x {ratio_vs_fa2:>8.2f}x")

    print()
    print("注：Naive/Ours 使用 float16，FA2 使用 float16（官方不支持 fp32）")
    print("    'vs FA2' 表示我们比 FA2 慢多少倍，>1 表示我们更慢")