#!/usr/bin/env python3
"""
Profiling harness for ncu (Nsight Compute).

Wraps each kernel launch in an NVTX range so ncu can filter by operator.
Run this via ncu_profile.sh, or directly:

  ncu --nvtx --nvtx-include "gemm" --set full -o out \\
      python kernels/profile.py --session ./sessions/2026-04-02 --op gemm

Usage:
  python kernels/profile.py --session ./sessions/2026-04-02
  python kernels/profile.py --session ./sessions/2026-04-02 --op gemm
  python kernels/profile.py --session ./sessions/2026-04-02 --op softmax --backend std
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

# Share data tables and loaders from run.py (no duplication).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as _run

WARMUP_ITERS = 3


def _cuda_warmup(device):
    """Force CUDA context + allocator init before any profiled region."""
    torch.cuda.init()
    x = torch.zeros(1, device=device)
    x.add_(1.0)
    torch.cuda.synchronize()
    del x


def _profile_op(op, backend, ext, session_dir, device, warmup=WARMUP_ITERS):
    """Warm up then do one NVTX-marked launch per bench case."""
    cases = _run.BENCH_CASES[op]

    for case in cases:
        label = case.get("label", _run._case_label(op, case))
        variant = case.get("variant", "layernorm")
        kw = {k: v for k, v in case.items() if k != "label"}
        inputs, _ = _run.REFS[op](device, **kw)

        def launch():
            if backend in ("std", "cuda"):
                _run._ext_forward(ext, op, inputs, variant)
            elif backend == "triton":
                _run._run_backend(op, "triton", inputs, session_dir, variant)
            else:
                _run._torch_forward(op, inputs, variant)

        # Warm-up before any profiler marker: ncu --profile-from-start no skips these.
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()

        # One profiled launch per case.
        # NVTX ranges appear in the GUI timeline for orientation.
        # cudaProfilerStart/Stop tell ncu exactly what to replay.
        nvtx.range_push(op)
        nvtx.range_push(label)
        torch.cuda.cudart().cudaProfilerStart()
        launch()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        nvtx.range_pop()
        nvtx.range_pop()

        print(f"  {label}")


def main():
    parser = argparse.ArgumentParser(
        description="ncu profiling harness — launch kernels with NVTX markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Via wrapper (recommended):
  ./kernels/ncu_profile.sh --session ./sessions/2026-04-02 --op gemm

  # Direct ncu invocation:
  ncu --nvtx --nvtx-include "gemm" --set full -o out \\
      python kernels/profile.py --session ./sessions/2026-04-02 --op gemm
""",
    )
    parser.add_argument(
        "--session",
        required=True,
        help="path to practice session directory",
    )
    parser.add_argument(
        "--op",
        choices=sorted(_run.REFS.keys()),
        default=None,
        help="operator to profile (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "std", "torch", "triton"],
        default="cuda",
        help="backend to profile (default: cuda)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP_ITERS,
        help=f"warm-up iterations before the profiled launch (default: {WARMUP_ITERS})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        sys.exit(1)

    warmup = args.warmup
    session_dir = Path(args.session).resolve()
    device = torch.device("cuda")
    ops = [args.op] if args.op else sorted(_run.REFS.keys())

    # Pre-load and compile the extension before any profiled launches.
    ext = None
    if args.backend == "cuda":
        print(f"Loading session extension ({session_dir.name}) ...")
        ext = _run._load_session_ext(session_dir)
    elif args.backend == "std":
        print("Loading std (reference) extension ...")
        ext = _run._load_std_ext()

    # Initialize CUDA context and allocator here, so that torch-internal
    # kernels don't appear inside the cudaProfilerStart/Stop region.
    _cuda_warmup(device)
    print()

    for op in ops:
        print(f"[{op}]")
        try:
            _profile_op(op, args.backend, ext, session_dir, device, warmup=warmup)
        except NotImplementedError:
            print("  skipped: not implemented")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
