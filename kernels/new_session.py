#!/usr/bin/env python3
"""
Initialize a new kernel practice session.

Usage:
  python new_session.py --out ./sessions/2026-04-01

Each session contains:
  *.cu            CUDA kernel skeletons  (implement launch_* functions)
  *_triton.py     Triton kernel skeletons
  bindings.cu     PyTorch C++ bindings   (auto-generated, do not edit)

To test your implementations, use the runner from kernels/:
  python run.py --session ./sessions/2026-04-01 --mode bench
  python run.py --session ./sessions/2026-04-01 --backend cuda --mode verify
"""

import argparse
from pathlib import Path

OPS = [
    "gemm",
    "softmax",
    "layernorm",
    "block_reduce",
    "flash_attention",
    "fused_mha",
]

# ── CUDA kernel skeletons ──────────────────────────────────────────────────────

CU_TEMPLATES = {
    "gemm": """\
#include <cuda_runtime.h>

// TODO: implement gemm kernel and launch.

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
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k; (void)stream;
}
""",
    "softmax": """\
#include <cuda_runtime.h>
#include <float.h>

// TODO: implement row-wise softmax.

__global__ void softmax_kernel(const float* x, float* y, int rows, int cols) {
    // TODO
    (void)x; (void)y; (void)rows; (void)cols;
}

extern "C" void launch_softmax(const float* x, float* y, int rows, int cols, cudaStream_t stream) {
    // TODO: configure grid/block and launch softmax_kernel
    (void)x; (void)y; (void)rows; (void)cols; (void)stream;
}
""",
    "layernorm": """\
#include <cuda_runtime.h>

// TODO: implement layernorm and rmsnorm.

__global__ void layernorm_kernel(
    const float* x, const float* gamma, const float* beta,
    float* y, int rows, int cols, float eps
) {
    // TODO
    (void)x; (void)gamma; (void)beta; (void)y; (void)rows; (void)cols; (void)eps;
}

__global__ void rmsnorm_kernel(
    const float* x, const float* gamma,
    float* y, int rows, int cols, float eps
) {
    // TODO
    (void)x; (void)gamma; (void)y; (void)rows; (void)cols; (void)eps;
}

extern "C" void launch_layernorm(
    const float* x, const float* gamma, const float* beta,
    float* y, int rows, int cols, float eps, cudaStream_t stream
) {
    // TODO
    (void)x; (void)gamma; (void)beta; (void)y; (void)rows; (void)cols; (void)eps; (void)stream;
}

extern "C" void launch_rmsnorm(
    const float* x, const float* gamma,
    float* y, int rows, int cols, float eps, cudaStream_t stream
) {
    // TODO
    (void)x; (void)gamma; (void)y; (void)rows; (void)cols; (void)eps; (void)stream;
}
""",
    "block_reduce": """\
#include <cuda_runtime.h>
#include <float.h>

// TODO: implement block reduce (sum).

__global__ void block_reduce_sum_kernel(const float* x, float* out, int n) {
    // TODO
    (void)x; (void)out; (void)n;
}

extern "C" void launch_block_reduce_sum(const float* x, float* out, int n, cudaStream_t stream) {
    // TODO
    (void)x; (void)out; (void)n; (void)stream;
}
""",
    "flash_attention": """\
#include <cuda_runtime.h>

// TODO: implement flash attention (4D tensors [B, H, N, d], causal mask).
//
// Grid:  x = ceil(N / BLOCK_Q),  y = B * H
// Block: BLOCK_Q threads
// Each blockIdx.y selects one (batch, head) slice; the slice base offset is
//   blockIdx.y * N * d  (tensors are contiguous in [B, H, N, d] layout).

__global__ void flash_attention_kernel(
    const float* q, const float* k, const float* v,
    float* o, int N, int d, float scale
) {
    // TODO
    (void)q; (void)k; (void)v; (void)o; (void)N; (void)d; (void)scale;
}

extern "C" void launch_flash_attention(
    const float* q, const float* k, const float* v,
    float* o, int B, int H, int N, int d, float scale, cudaStream_t stream
) {
    // TODO: set grid = (ceil(N/BLOCK_Q), B*H), block = BLOCK_Q
    (void)q; (void)k; (void)v; (void)o; (void)B; (void)H; (void)N; (void)d; (void)scale; (void)stream;
}
""",
    "fused_mha": """\
#include <cuda_runtime.h>

// TODO: implement naive fused MHA (materialize scores per query).

__global__ void fused_mha_kernel(
    const float* q, const float* k, const float* v,
    float* o, int num_q, int n, int d, float scale
) {
    // TODO
    (void)q; (void)k; (void)v; (void)o; (void)num_q; (void)n; (void)d; (void)scale;
}

extern "C" void launch_fused_mha(
    const float* q, const float* k, const float* v,
    float* o, int num_q, int n, int d, float scale, cudaStream_t stream
) {
    // TODO
    (void)q; (void)k; (void)v; (void)o; (void)num_q; (void)n; (void)d; (void)scale; (void)stream;
}
""",
}

# ── Triton skeletons ───────────────────────────────────────────────────────────

TRITON_TEMPLATES = {
    "gemm": """\
import torch
import triton
import triton.language as tl


# TODO: implement a Triton GEMM kernel.

def forward(a, b):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement gemm triton kernel")
""",
    "softmax": """\
import torch
import triton
import triton.language as tl

# TODO: implement row-wise softmax.

def forward(x):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement softmax triton kernel")
""",
    "layernorm": """\
import torch
import triton
import triton.language as tl

# TODO: implement layernorm and rmsnorm. Use variant to select.

def forward(x, gamma, beta=None, variant="layernorm", eps=1e-5):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement layernorm/rmsnorm triton kernel")
""",
    "block_reduce": """\
import torch
import triton
import triton.language as tl

# TODO: implement block reduce (sum).

def forward(x):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement block reduce triton kernel")
""",
    "flash_attention": """\
import torch
import triton
import triton.language as tl

# TODO: implement flash attention (single head, causal).

def forward(q, k, v):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement flash attention triton kernel")
""",
    "fused_mha": """\
import torch
import triton
import triton.language as tl

# TODO: implement fused MHA (naive, materialize scores).

def forward(q, k, v):
    if triton is None:
        raise RuntimeError("triton is not available")
    raise NotImplementedError("implement fused MHA triton kernel")
""",
}

# ── Session bindings (auto-generated, wraps launch_* for PyTorch extension) ────

BINDINGS_CU = """\
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include "gemm.cu"
#include "softmax.cu"
#include "layernorm.cu"
#include "block_reduce.cu"
#include "flash_attention.cu"
#include "fused_mha.cu"

#define CHECK_CUDA(x)       TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x)      TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline void check_cuda_err() {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
}

static inline cudaStream_t get_stream() {
    return at::cuda::getDefaultCUDAStream().stream();
}

torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a); CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
    int m = (int)a.size(0), k = (int)a.size(1), n = (int)b.size(1);
    TORCH_CHECK(k == (int)b.size(0), "gemm: inner dims mismatch");
    auto c = torch::zeros({m, n}, a.options());
    launch_gemm(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), m, n, k, get_stream());
    check_cuda_err();
    return c;
}

torch::Tensor softmax(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "softmax expects 2D tensor");
    auto out = torch::empty_like(x);
    launch_softmax(x.data_ptr<float>(), out.data_ptr<float>(),
                   (int)x.size(0), (int)x.size(1), get_stream());
    check_cuda_err();
    return out;
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    TORCH_CHECK(x.dim() == 2, "layernorm expects 2D tensor");
    int rows = (int)x.size(0), cols = (int)x.size(1);
    TORCH_CHECK((int)gamma.size(0) == cols && (int)beta.size(0) == cols, "gamma/beta shape mismatch");
    auto out = torch::empty_like(x);
    launch_layernorm(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                     out.data_ptr<float>(), rows, cols, (float)eps, get_stream());
    check_cuda_err();
    return out;
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor gamma, double eps) {
    CHECK_INPUT(x); CHECK_INPUT(gamma);
    TORCH_CHECK(x.dim() == 2, "rmsnorm expects 2D tensor");
    int rows = (int)x.size(0), cols = (int)x.size(1);
    TORCH_CHECK((int)gamma.size(0) == cols, "gamma shape mismatch");
    auto out = torch::empty_like(x);
    launch_rmsnorm(x.data_ptr<float>(), gamma.data_ptr<float>(),
                   out.data_ptr<float>(), rows, cols, (float)eps, get_stream());
    check_cuda_err();
    return out;
}

torch::Tensor block_reduce_sum(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1, "block_reduce_sum expects 1D tensor");
    auto out = torch::zeros({}, x.options());
    launch_block_reduce_sum(x.data_ptr<float>(), out.data_ptr<float>(), (int)x.size(0), get_stream());
    check_cuda_err();
    return out;
}

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "flash_attention expects 4D tensors [B, H, N, d]");
    TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes(), "K/V shape mismatch");
    int B = (int)q.size(0), H = (int)q.size(1), N = (int)q.size(2), d = (int)q.size(3);
    float scale = 1.0f / std::sqrt((float)d);
    auto out = torch::zeros_like(q);
    launch_flash_attention(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                           out.data_ptr<float>(), B, H, N, d, scale, get_stream());
    check_cuda_err();
    return out;
}

torch::Tensor fused_mha(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "fused_mha expects 2D tensors");
    int num_q = (int)q.size(0), d = (int)q.size(1), n = (int)k.size(0);
    float scale = 1.0f / std::sqrt((float)d);
    auto out = torch::zeros({num_q, d}, q.options());
    launch_fused_mha(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                     out.data_ptr<float>(), num_q, n, d, scale, get_stream());
    check_cuda_err();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm",             &gemm,            "GEMM");
    m.def("softmax",          &softmax,          "Softmax");
    m.def("layernorm",        &layernorm,        "LayerNorm");
    m.def("rmsnorm",          &rmsnorm,          "RMSNorm");
    m.def("block_reduce_sum", &block_reduce_sum, "Block reduce sum");
    m.def("flash_attention",  &flash_attention,  "Flash attention");
    m.def("fused_mha",        &fused_mha,        "Fused MHA");
}
"""


def write_file(path, content, force=False):
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists")
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True, help="output folder for the session")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # kernel skeletons + triton skeletons
    for op in OPS:
        write_file(out_dir / f"{op}.cu", CU_TEMPLATES[op], force=args.force)
        write_file(out_dir / f"{op}_triton.py", TRITON_TEMPLATES[op], force=args.force)

    # bindings (auto-generated, wires launch_* into the PyTorch extension)
    write_file(out_dir / "bindings.cu", BINDINGS_CU, force=args.force)

    print(f"initialized session in {out_dir}")
    print()
    print("run from kernels/:")
    print(f"  python run.py --session {out_dir} --mode bench")
    print(f"  python run.py --session {out_dir} --backend cuda --mode verify")


if __name__ == "__main__":
    main()
