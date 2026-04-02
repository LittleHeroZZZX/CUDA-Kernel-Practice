#!/usr/bin/env python3
"""
Standalone kernel verify / benchmark harness.

Usage (from kernels/ or anywhere):
  python run.py --session ./sessions/2026-03-31
  python run.py --session ./sessions/2026-03-31 --op gemm --mode bench
  python run.py --mode bench          # torch + std only, no session required
"""

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# ── Paths ──────────────────────────────────────────────────────────────────────

PRACTICE_DIR = Path(__file__).resolve().parent
KERNELS_STD = PRACTICE_DIR / "reference"

# ── ANSI helpers ───────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty()


def _ansi(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def green(s):
    return _ansi("32", s)


def red(s):
    return _ansi("31", s)


def yellow(s):
    return _ansi("33", s)


def bold(s):
    return _ansi("1", s)


def dim(s):
    return _ansi("2", s)


# ── Utilities ──────────────────────────────────────────────────────────────────


def get_device(prefer="cuda"):
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bench(fn, warmup=5, iters=20, rounds=5):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        times = []
        for _ in range(rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / iters)
        times.sort()
        return times[rounds // 2]
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - t0) * 1000.0 / iters)
    times.sort()
    return times[rounds // 2]


def check_close(a, b, rtol=1e-3, atol=1e-3):
    if isinstance(a, (tuple, list)):
        return all(check_close(x, y, rtol, atol) for x, y in zip(a, b))
    return torch.allclose(a, b, rtol=rtol, atol=atol)


# ── Extension loaders ──────────────────────────────────────────────────────────

_STD_EXT = None
_SESSION_EXT = None


def _load_std_ext():
    global _STD_EXT
    if _STD_EXT is not None:
        return _STD_EXT
    build = KERNELS_STD / ".build"
    build.mkdir(exist_ok=True)
    _STD_EXT = load(
        name="llm_kernels_std",
        sources=[str(KERNELS_STD / "bindings.cu")],
        extra_include_paths=[str(KERNELS_STD)],
        extra_cuda_cflags=["--use_fast_math"],
        with_cuda=True,
        verbose=False,
        build_directory=str(build),
    )
    return _STD_EXT


def _load_session_ext(session_dir):
    global _SESSION_EXT
    if _SESSION_EXT is not None:
        return _SESSION_EXT
    bindings = session_dir / "bindings.cu"
    if not bindings.exists():
        raise RuntimeError(
            f"No bindings.cu found in {session_dir}. "
            "Re-run init_kernels.py to regenerate the session."
        )
    name = "llm_kernels_session_" + session_dir.name.replace("-", "_")
    build = session_dir / ".build"
    build.mkdir(exist_ok=True)
    _SESSION_EXT = load(
        name=name,
        sources=[str(bindings)],
        extra_include_paths=[str(session_dir)],
        extra_cuda_cflags=["--use_fast_math"],
        with_cuda=True,
        verbose=False,
        build_directory=str(build),
    )
    return _SESSION_EXT


def _import_from_session(module_name, session_dir):
    path = session_dir / f"{module_name}.py"
    if not path.exists():
        raise RuntimeError(f"{path} not found")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Reference generators ───────────────────────────────────────────────────────


def ref_gemm(device, m=256, n=256, k=256):
    a = torch.randn((m, k), device=device, dtype=torch.float32)
    b = torch.randn((k, n), device=device, dtype=torch.float32)
    return (a, b), torch.matmul(a, b)


def ref_softmax(device, rows=128, cols=256):
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    return (x,), torch.softmax(x, dim=-1)


def ref_layernorm(device, rows=128, cols=256, eps=1e-5, variant="layernorm"):
    x = torch.randn((rows, cols), device=device, dtype=torch.float32)
    gamma = torch.randn((cols,), device=device, dtype=torch.float32)
    if variant == "rmsnorm":
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return (x, gamma), x / rms * gamma
    beta = torch.randn((cols,), device=device, dtype=torch.float32)
    return (x, gamma, beta), F.layer_norm(x, (cols,), gamma, beta, eps)


def ref_block_reduce(device, n=1 << 20):
    x = torch.randn((n,), device=device, dtype=torch.float32)
    return (x,), torch.sum(x)


def ref_flash_attention(device, n=256, d=64):
    q, k, v = [
        torch.randn((n, d), device=device, dtype=torch.float32) for _ in range(3)
    ]
    scale = 1.0 / (d**0.5)
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    mask = torch.tril(torch.ones((n, n), device=device, dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return (q, k, v), torch.matmul(torch.softmax(scores, dim=-1), v)


def ref_fused_mha(device, num_q=8, n=256, d=64):
    q = torch.randn((num_q, d), device=device, dtype=torch.float32)
    k = torch.randn((n, d), device=device, dtype=torch.float32)
    v = torch.randn((n, d), device=device, dtype=torch.float32)
    scale = 1.0 / (d**0.5)
    scores = torch.matmul(q, k.transpose(0, 1)) * scale
    return (q, k, v), torch.matmul(torch.softmax(scores, dim=-1), v)


REFS = {
    "gemm": ref_gemm,
    "softmax": ref_softmax,
    "layernorm": ref_layernorm,
    "block_reduce": ref_block_reduce,
    "flash_attention": ref_flash_attention,
    "fused_mha": ref_fused_mha,
}

BACKENDS = ["torch", "std", "triton", "cuda"]

# ── Test cases ─────────────────────────────────────────────────────────────────
#
# VERIFY_CASES: multiple shapes per op — covers basic, non-power-of-2, and
#               tile-boundary sizes to stress-test tiling and edge handling.
#
# BENCH_CASES:  LLM-scale shapes representative of transformer workloads.
#               Reference: LLaMA-7B (hidden=4096, heads=32, head_dim=128,
#               ffn_intermediate=11008, typical seq_len=2048).
#               Each case has a human-readable "label" for display.

VERIFY_CASES = {
    "gemm": [
        dict(m=64, n=64, k=64),  # fits in one tile block
        dict(m=128, n=128, k=128),  # basic square
        dict(m=256, n=512, k=128),  # non-square
        dict(m=100, n=200, k=300),  # non-power-of-2
        dict(m=65, n=65, k=17),  # tile-boundary +1  (TILE_M=64, TILE_K=16)
    ],
    "softmax": [
        dict(rows=1, cols=128),  # single row
        dict(rows=128, cols=256),  # basic
        dict(rows=64, cols=1024),  # wide cols
        dict(rows=256, cols=33),  # odd cols (not warp-aligned)
        dict(rows=2048, cols=128),  # many rows, narrow
    ],
    "layernorm": [
        dict(rows=128, cols=256, variant="layernorm"),
        dict(rows=64, cols=1024, variant="layernorm"),
        dict(rows=128, cols=300, variant="layernorm"),  # non-power-of-2
        dict(rows=128, cols=256, variant="rmsnorm"),
        dict(rows=64, cols=1024, variant="rmsnorm"),
    ],
    "block_reduce": [
        dict(n=256),
        dict(n=1024),
        dict(n=1025),  # crosses single-block boundary
        dict(n=1 << 20),  # 1M elements
        dict(n=100003),  # non-power-of-2
    ],
    "flash_attention": [
        dict(n=64, d=32),
        dict(n=128, d=64),
        dict(n=256, d=64),
        dict(n=128, d=128),
        dict(n=257, d=64),  # non-power-of-2 seq
    ],
    "fused_mha": [
        dict(num_q=1, n=64, d=32),  # single-query decode
        dict(num_q=8, n=128, d=64),
        dict(num_q=16, n=256, d=64),
        dict(num_q=4, n=257, d=64),  # non-power-of-2 KV
    ],
}

BENCH_CASES = {
    "gemm": [
        # weight matrix multiply: (tokens, hidden) × (hidden, hidden)
        dict(m=4096, n=4096, k=4096, label="4096 × 4096 × 4096   attn proj"),
        # FFN up/gate: (tokens, hidden) × (hidden, ffn_intermediate)
        dict(m=4096, n=11008, k=4096, label="4096 × 11008 × 4096  ffn up"),
        # FFN down: (tokens, ffn_intermediate) × (ffn_intermediate, hidden)
        dict(m=4096, n=4096, k=11008, label="4096 × 4096 × 11008  ffn down"),
    ],
    "softmax": [
        # attention score matrix per head: seq_len × seq_len
        dict(rows=2048, cols=2048, label="2048 × 2048  seq=2048 attn scores"),
    ],
    "layernorm": [
        # LLaMA-style: RMSNorm before every attention/FFN block
        dict(rows=2048, cols=4096, variant="rmsnorm", label="rmsnorm   2048 × 4096"),
        dict(rows=2048, cols=4096, variant="layernorm", label="layernorm 2048 × 4096"),
    ],
    "block_reduce": [
        # reduce over full hidden states buffer: seq_len × hidden
        dict(n=2048 * 4096, label="n = 8M  (2048 × 4096)"),
    ],
    "flash_attention": [
        # single-head self-attention at seq=2048, head_dim=128
        dict(n=2048, d=128, label="seq=2048  d=128"),
        dict(n=4096, d=128, label="seq=4096  d=128  4K context"),
    ],
    "fused_mha": [
        # single-token decode: 1 query attending to full KV cache
        dict(num_q=1, n=2048, d=128, label="q=1   kv=2048  d=128  decode"),
        # speculative decode / prefill chunk
        dict(num_q=64, n=2048, d=128, label="q=64  kv=2048  d=128  prefill"),
    ],
}


def _case_label(op, case):
    """Compact shape label for verify output."""
    if op == "gemm":
        return f"{case['m']}×{case['n']}×{case['k']}"
    if op == "softmax":
        return f"{case['rows']}×{case['cols']}"
    if op == "layernorm":
        v = "rms" if case.get("variant") == "rmsnorm" else "ln"
        return f"{case['rows']}×{case['cols']} ({v})"
    if op == "block_reduce":
        n = case["n"]
        return f"n={n:,}"
    if op == "flash_attention":
        return f"n={case['n']} d={case['d']}"
    if op == "fused_mha":
        return f"q={case['num_q']} kv={case['n']} d={case['d']}"
    return str(case)


# ── Backend dispatch ───────────────────────────────────────────────────────────


def _torch_forward(op, inputs, variant="layernorm"):
    if op == "gemm":
        a, b = inputs
        return torch.matmul(a, b)
    if op == "softmax":
        (x,) = inputs
        return torch.softmax(x, dim=-1)
    if op == "layernorm":
        if variant == "rmsnorm":
            x, gamma = inputs
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
            return x / rms * gamma
        x, gamma, beta = inputs
        return F.layer_norm(x, (x.size(-1),), gamma, beta, 1e-5)
    if op == "block_reduce":
        (x,) = inputs
        return torch.sum(x)
    if op == "flash_attention":
        q, k, v = inputs
        d = q.size(-1)
        scale = 1.0 / (d**0.5)
        scores = torch.matmul(q, k.transpose(0, 1)) * scale
        mask = torch.tril(torch.ones_like(scores, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v)
    if op == "fused_mha":
        q, k, v = inputs
        d = q.size(-1)
        scale = 1.0 / (d**0.5)
        scores = torch.matmul(q, k.transpose(0, 1)) * scale
        return torch.matmul(torch.softmax(scores, dim=-1), v)
    raise ValueError(f"unknown op: {op}")


def _ext_forward(ext, op, inputs, variant="layernorm"):
    if op == "gemm":
        a, b = inputs
        return ext.gemm(a, b)
    if op == "softmax":
        (x,) = inputs
        return ext.softmax(x)
    if op == "layernorm":
        if variant == "rmsnorm":
            x, gamma = inputs
            return ext.rmsnorm(x, gamma, 1e-5)
        x, gamma, beta = inputs
        return ext.layernorm(x, gamma, beta, 1e-5)
    if op == "block_reduce":
        (x,) = inputs
        return ext.block_reduce_sum(x)
    if op == "flash_attention":
        q, k, v = inputs
        return ext.flash_attention(q, k, v)
    if op == "fused_mha":
        q, k, v = inputs
        return ext.fused_mha(q, k, v)
    raise ValueError(f"unknown op: {op}")


def _run_backend(op, backend, inputs, session_dir, variant="layernorm"):
    if backend == "torch":
        return _torch_forward(op, inputs, variant)
    if backend == "std":
        return _ext_forward(_load_std_ext(), op, inputs, variant)
    if backend == "triton":
        if session_dir is None:
            raise RuntimeError("--session required for triton backend")
        mod = _import_from_session(f"{op}_triton", session_dir)
        if op == "layernorm":
            return mod.forward(*inputs, variant=variant)
        return mod.forward(*inputs)
    if backend == "cuda":
        if session_dir is None:
            raise RuntimeError("--session required for cuda backend")
        return _ext_forward(_load_session_ext(session_dir), op, inputs, variant)
    raise ValueError(f"unknown backend: {backend}")


# ── Display ────────────────────────────────────────────────────────────────────


def _fmt_ms(ms):
    if ms < 0.01:
        return f"{ms * 1000:.1f} µs"
    if ms < 1.0:
        return f"{ms:.3f} ms"
    return f"{ms:.2f} ms"


# ── Verify mode ────────────────────────────────────────────────────────────────


def _run_verify(ops, backend, device, session_dir, show_traceback=False):
    print(f"\n{bold('Verify')}  backend={backend}  device={device.type}\n")

    W_op = max(len(op) for op in ops)
    W_label = 28
    total_pass = total_fail = 0

    for op in ops:
        cases = VERIFY_CASES[op]
        print(f"  {bold(op)}")

        op_pass = op_fail = 0
        for case in cases:
            label = _case_label(op, case)
            variant = case.get("variant", "layernorm")

            if backend in ("cuda", "triton", "std") and device.type != "cuda":
                tag = yellow("SKIP") + dim("  cuda required")
                op_fail += 1
            else:
                try:
                    kw = {k: v for k, v in case.items() if k != "label"}
                    inputs, ref = REFS[op](device, **kw)
                    out = _run_backend(op, backend, inputs, session_dir, variant)
                    if check_close(out, ref):
                        tag = green("PASS")
                        op_pass += 1
                    else:
                        tag = red("FAIL") + dim("  values differ")
                        op_fail += 1
                except NotImplementedError as exc:
                    msg = str(exc).splitlines()[0][:40] or "not implemented"
                    tag = yellow("TODO") + dim(f"  {msg}")
                    op_fail += 1
                except Exception as exc:
                    if show_traceback:
                        tb = traceback.format_exc().rstrip().splitlines()
                        for line in tb:
                            print(dim(f"      {line}"))
                    msg = str(exc).splitlines()[0][:50]
                    tag = red("ERR ") + dim(f"  {msg}")
                    op_fail += 1

            print(f"    {label:<{W_label}}  {tag}")

        total_pass += op_pass
        total_fail += op_fail
        n = op_pass + op_fail
        summary = green(f"{op_pass}/{n}") if op_fail == 0 else red(f"{op_pass}/{n}")
        print(f"    {dim('─' * W_label)}  {summary}")
        print()

    grand_total = total_pass + total_fail
    if total_fail == 0:
        print(f"  {green(f'All {grand_total} passed.')}\n")
    else:
        print(
            f"  {green(f'{total_pass} passed')}, {red(f'{total_fail} failed')}  "
            f"{dim(f'({grand_total} total)')}\n"
        )

    if total_fail:
        raise SystemExit(1)


# ── Bench mode ─────────────────────────────────────────────────────────────────


def _run_bench(ops, backends, device, session_dir, variant_filter, show_traceback=False):
    print(f"\n{bold('Bench')}  device={device.type}\n")

    # Build the flat list of (op, case) rows, applying variant filter
    rows = []
    for op in ops:
        for case in BENCH_CASES[op]:
            if op == "layernorm" and variant_filter:
                if case.get("variant", "layernorm") != variant_filter:
                    continue
            rows.append((op, case))

    if not rows:
        print("  (no cases to run)\n")
        return

    W_row = max(len(f"{op}  {c['label']}") for op, c in rows) + 2
    W_t = 22

    hdr = f"  {'op  [shape]':<{W_row}}" + "".join(f"{b:<{W_t}}" for b in backends)
    print(dim(hdr))
    print(dim("  " + "─" * (W_row + W_t * len(backends))))

    any_error = False

    for op, case in rows:
        label = case["label"]
        variant = case.get("variant", "layernorm")
        row_key = f"{op}  {label}"

        kw = {k: v for k, v in case.items() if k not in ("label",)}
        inputs, _ = REFS[op](device, **kw)

        times = {}
        for b in backends:
            if b in ("cuda", "triton", "std") and device.type != "cuda":
                times[b] = ("skip", None)
                continue
            try:
                ms = bench(
                    lambda _b=b: _run_backend(op, _b, inputs, session_dir, variant)
                )
                times[b] = ("ok", ms)
            except NotImplementedError:
                times[b] = ("notimpl", None)
            except Exception as exc:
                if show_traceback:
                    tb = traceback.format_exc().rstrip().splitlines()
                    for line in tb:
                        print(dim(f"  {line}"))
                times[b] = ("error", str(exc).splitlines()[0][:30])
                any_error = True

        torch_ms = times.get("torch", (None, None))[1]

        cells = []
        for b in backends:
            kind, val = times[b]
            if kind == "ok":
                s = _fmt_ms(val)
                if b != "torch" and torch_ms is not None:
                    s += f"  ({torch_ms / val:.1f}x)"
                padded = f"{s:<{W_t}}"
                if b != "torch" and torch_ms is not None:
                    if val < torch_ms:
                        cells.append(green(padded))
                    elif val > torch_ms:
                        cells.append(red(padded))
                    else:
                        cells.append(padded)
                else:
                    cells.append(padded)
            elif kind == "notimpl":
                cells.append(dim(f"{'--':<{W_t}}"))
            elif kind == "skip":
                cells.append(dim(f"{'skip':<{W_t}}"))
            else:
                cells.append(red(f"{'error':<{W_t}}"))

        print(f"  {row_key:<{W_row}}" + "".join(cells))

    print()
    if "torch" in backends and len(backends) > 1:
        print(
            f"  {dim('Speedup relative to torch  (green = faster, red = slower).')}\n"
        )

    if any_error:
        raise SystemExit(1)


# ── Entry point ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Kernel verify / benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py --session ./sessions/2026-03-31 --op gemm --mode bench
  python run.py --session ./sessions/2026-03-31 --backend cuda
  python run.py --mode bench          # torch + std only (no session needed)
  python run.py --mode bench --op layernorm --variant rmsnorm
""",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="path to practice session directory (required for triton/cuda backends)",
    )
    parser.add_argument("--op", choices=sorted(REFS.keys()), default=None)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="backend  [verify: default=torch; bench: default=all available]",
    )
    parser.add_argument("--mode", choices=["verify", "bench"], default="verify")
    parser.add_argument(
        "--variant",
        choices=["layernorm", "rmsnorm"],
        default=None,
        help="filter layernorm bench cases (default: show both)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print full tracebacks for errors",
    )
    args = parser.parse_args()

    session_dir = Path(args.session).resolve() if args.session else None
    device = get_device(args.device)
    ops = [args.op] if args.op else sorted(REFS.keys())

    if args.mode == "verify":
        backend = args.backend or "torch"
        _run_verify(ops, backend, device, session_dir, args.verbose)
    else:
        if args.backend:
            backends = [args.backend]
        elif session_dir is not None:
            backends = BACKENDS
        else:
            backends = ["torch", "std"]
        _run_bench(ops, backends, device, session_dir, args.variant, args.verbose)


if __name__ == "__main__":
    main()
