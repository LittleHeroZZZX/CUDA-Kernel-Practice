# CUDA Kernel Practice &nbsp;·&nbsp; [中文](README_CN.md)

A structured environment for implementing and benchmarking GPU kernels for LLM inference from scratch.

Six core operators are covered, each with a reference implementation, skeleton files, and a test harness that verifies correctness and measures speed against PyTorch.

---

## Operators

| Operator | Signature | Notes |
| --- | --- | --- |
| **GEMM** | `C[M,N] = A[M,K] @ B[K,N]` | 2-level tiling: 64×64 block tile, 4×4 register tile |
| **Softmax** | row-wise, numerically stable | 2-pass: max then exp/sum |
| **LayerNorm** | `(x - μ) / σ * γ + β` | 2-pass mean & variance |
| **RMSNorm** | `x / rms(x) * γ` | single-pass, used in LLaMA |
| **Block Reduce** | sum or max over arbitrary arrays | 2-kernel pattern: partial → final |
| **Flash Attention** | single-head, causal or full | online softmax with blocked K/V |
| **Fused MHA** | multi-head decode / prefill | fused QKV projection + attention |

---

## Requirements

- CUDA toolkit ≥ 11.8 (nvcc, cuda_runtime.h)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

---

## Quick Start

### 1. Clone and set up

```bash
git clone <repo-url>
cd cudaEnv
uv sync
source .venv/bin/activate
```

`uv sync` reads [pyproject.toml](pyproject.toml) and installs PyTorch (CUDA 12.8 wheel) and Triton into `.venv` automatically.

To target a different CUDA version, edit the index URL in `pyproject.toml` (e.g. `cu121`, `cu124`) before running `uv sync`.

### 2. Create a practice session

```bash
python kernels/new_session.py --out kernels/sessions/$(date +%Y-%m-%d)
```

This generates skeleton files for all six operators:

```text
kernels/sessions/2026-04-02/
├── gemm.cu                # CUDA skeleton — implement this
├── softmax.cu
├── layernorm.cu
├── block_reduce.cu
├── flash_attention.cu
├── fused_mha.cu
├── gemm_triton.py         # Triton skeleton — or this
├── softmax_triton.py
├── ...
└── bindings.cu            # auto-generated PyTorch C++ bindings
```

### 3. Implement a kernel

Open `gemm.cu` (or any operator file). You will find a kernel stub and a `launch_*` function:

```c
__global__ void gemm_kernel(
    const float* a, const float* b, float* c,
    int m, int n, int k
) {
    // TODO: compute c[m, n] = a[m, k] * b[k, n]
}

extern "C" void launch_gemm(
    const float* a, const float* b, float* c,
    int m, int n, int k, cudaStream_t stream
) {
    // TODO: configure grid/block and launch gemm_kernel
}
```

Fill in the kernel logic. The `launch_*` convention is what `bindings.cu` calls, so keep the signature.

If you prefer Triton, edit the corresponding `*_triton.py` instead.

### 4. Verify correctness

```bash
# verify one operator against torch reference
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --op gemm --backend cuda --mode verify

# verify all operators at once
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --backend cuda --mode verify
```

Output shows PASS / FAIL / ERR for each test shape:

```text
gemm  [64, 64, 64]        PASS
gemm  [128, 256, 512]     PASS
gemm  [1000, 1000, 1000]  FAIL  max_err=0.012
```

### 5. Benchmark

```bash
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --op gemm --backend cuda --mode bench
```

Compares your kernel against torch (cuBLAS) on LLM-scale shapes:

```text
gemm  [4096, 4096, 4096]   cuda 3.21 ms   torch 1.05 ms   0.33x
gemm  [4096, 11008, 4096]  cuda 8.42 ms   torch 2.81 ms   0.33x
```

Triton backend works the same way — just swap `--backend triton`.

---

## Backends

| Backend | What it runs | When to use |
| --- | --- | --- |
| `torch` | PyTorch ops (cuBLAS etc.) | reference / speed ceiling |
| `std` | `kernels/reference/` reference CUDA | see a working implementation |
| `cuda` | your session `*.cu` files | your implementation |
| `triton` | your session `*_triton.py` files | your Triton implementation |

Run `--backend torch --mode bench` without a session to get reference timings on your GPU.

---

## Project Structure

```text
cudaEnv/
├── kernels/
│   ├── run.py                    # verify / benchmark harness
│   ├── new_session.py            # session generator
│   ├── llm_kernels.cu            # all 6 operators in one file (well-commented)
│   ├── reference/                # split reference implementations (one file per op)
│   │   ├── common.cuh            # shared warp/block reduction helpers
│   │   ├── gemm.cu
│   │   ├── softmax.cu
│   │   ├── layernorm.cu
│   │   ├── block_reduce.cu
│   │   ├── flash_attention.cu
│   │   ├── fused_mha.cu
│   │   └── bindings.cu
│   ├── sessions/
│   │   └── YYYY-MM-DD/           # one folder per practice day
│   │       ├── *.cu
│   │       ├── *_triton.py
│   │       └── bindings.cu
│   └── lessons.md                # pitfalls collected during development
├── examples/
│   ├── gemm/                     # step-by-step GEMM optimization walkthrough
│   │   ├── 01_gemm_native.cu     # naive baseline (one thread per output element)
│   │   ├── 02_gemm_tile.cu       # shared memory tiling
│   │   ├── 03_gemm_reg.cu        # register tiling
│   │   └── benchmark.py
│   └── flash_attention/          # standalone flash attention experiments
│       ├── naive_attention.py
│       ├── flash_attention.py
│       └── benchmark.py
├── pyproject.toml
├── README.md
└── README_CN.md
```

### Reading order

1. **`kernels/llm_kernels.cu`** — start here. All six operators in one file with detailed comments explaining the tiling strategy, warp/block reduction pattern, and online softmax math.
2. **`kernels/reference/`** — same implementations split into separate files. Good for referencing one operator at a time after you have read the monolithic version.
3. **`examples/gemm/`** — if GEMM tiling is new, `01_gemm_native.cu` shows the simplest possible kernel before introducing shared memory and register tiling.

---

## run.py Reference

```text
python kernels/run.py [options]

  --session PATH      path to a session folder (required for cuda/triton backends)
  --op OP             operator to test: gemm | softmax | layernorm | block_reduce
                      | flash_attention | fused_mha  (default: all)
  --variant VARIANT   rmsnorm (for layernorm op), decode | prefill (for fused_mha)
  --backend BACKEND   torch | std | cuda | triton  (default: all available)
  --mode MODE         verify | bench  (default: verify)
```

---

## Common Pitfalls

A few sharp edges that come up repeatedly when implementing these kernels:

- **Shared memory writes need `__syncthreads()`** before any thread reads what another thread wrote.
- **Block reduce phase 2**: after `__syncthreads()`, re-read from shared memory into a register — do not reuse the register value from phase 1.
- **`bindings.cu` includes all `.cu` files in the same translation unit.** Helper `__device__` functions must have unique names across operators (use a prefix like `ln_`, `sm_`, `fa_`).
- **LayerNorm mean**: index with the row pointer, not the global pointer. `x[i]` always reads from row 0 if `x` is not offset.
- **Variance denominator**: divide by `cols` (number of elements), not by the raw reduction sum.

---

## License

MIT
