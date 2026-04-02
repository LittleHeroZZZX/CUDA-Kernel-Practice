#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <vector>

#include "gemm.cu"
#include "softmax.cu"
#include "layernorm.cu"
#include "block_reduce.cu"
#include "flash_attention.cu"
#include "fused_mha.cu"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

static inline void check_cuda(cudaError_t err) {
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

static inline cudaStream_t get_stream() {
    return at::cuda::getDefaultCUDAStream().stream();
}

torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2D tensors");
    int M = static_cast<int>(a.size(0));
    int K = static_cast<int>(a.size(1));
    int K2 = static_cast<int>(b.size(0));
    int N = static_cast<int>(b.size(1));
    TORCH_CHECK(K == K2, "gemm shapes mismatch");

    auto c = torch::zeros({M, N}, a.options());
    dim3 block(TILE_N / WN, TILE_M / WM);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    gemm_kernel<<<grid, block, 0, get_stream()>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    check_cuda(cudaGetLastError());
    return c;
}

torch::Tensor softmax(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "softmax expects 2D tensor");
    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));

    auto out = torch::empty_like(x);
    int threads = std::min(cols, 1024);
    softmax_kernel<<<rows, threads, 0, get_stream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), cols);
    check_cuda(cudaGetLastError());
    return out;
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    TORCH_CHECK(x.dim() == 2, "layernorm expects 2D tensor");
    TORCH_CHECK(gamma.dim() == 1 && beta.dim() == 1, "gamma/beta must be 1D");

    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));
    TORCH_CHECK(static_cast<int>(gamma.size(0)) == cols, "gamma shape mismatch");
    TORCH_CHECK(static_cast<int>(beta.size(0)) == cols, "beta shape mismatch");

    auto out = torch::empty_like(x);
    int threads = std::min(cols, 1024);
    layernorm_kernel<<<rows, threads, 0, get_stream()>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(), cols, static_cast<float>(eps));
    check_cuda(cudaGetLastError());
    return out;
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor gamma, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    TORCH_CHECK(x.dim() == 2, "rmsnorm expects 2D tensor");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");

    int rows = static_cast<int>(x.size(0));
    int cols = static_cast<int>(x.size(1));
    TORCH_CHECK(static_cast<int>(gamma.size(0)) == cols, "gamma shape mismatch");

    auto out = torch::empty_like(x);
    int threads = std::min(cols, 1024);
    rmsnorm_kernel<<<rows, threads, 0, get_stream()>>>(
        x.data_ptr<float>(), gamma.data_ptr<float>(), out.data_ptr<float>(),
        cols, static_cast<float>(eps));
    check_cuda(cudaGetLastError());
    return out;
}

torch::Tensor block_reduce_sum_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 1, "block_reduce_sum expects 1D tensor");
    int N = static_cast<int>(x.size(0));

    int threads = 256;
    int blocks = std::min((N + threads - 1) / threads, 1024);
    auto partial = torch::zeros({blocks}, x.options());
    auto out = torch::zeros({}, x.options());

    block_reduce_sum_kernel<<<blocks, threads, 0, get_stream()>>>(
        x.data_ptr<float>(), partial.data_ptr<float>(), N);
    check_cuda(cudaGetLastError());

    final_reduce_sum_kernel<<<1, threads, 0, get_stream()>>>(
        partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
    check_cuda(cudaGetLastError());

    return out;
}

torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "flash_attention expects 2D tensors");

    int N = static_cast<int>(q.size(0));
    int d = static_cast<int>(q.size(1));
    TORCH_CHECK(static_cast<int>(k.size(0)) == N && static_cast<int>(v.size(0)) == N, "K/V shape mismatch");
    TORCH_CHECK(static_cast<int>(k.size(1)) == d && static_cast<int>(v.size(1)) == d, "K/V dim mismatch");
    TORCH_CHECK(d <= MAX_D, "d exceeds MAX_D");

    auto out = torch::zeros_like(q);
    dim3 block(BLOCK_Q);
    dim3 grid((N + BLOCK_Q - 1) / BLOCK_Q);
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    flash_attention_kernel<<<grid, block, 0, get_stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(), N, d, scale);
    check_cuda(cudaGetLastError());
    return out;
}

torch::Tensor fused_mha(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2 && v.dim() == 2, "fused_mha expects 2D tensors");

    int num_q = static_cast<int>(q.size(0));
    int d = static_cast<int>(q.size(1));
    int N = static_cast<int>(k.size(0));
    TORCH_CHECK(static_cast<int>(k.size(1)) == d && static_cast<int>(v.size(1)) == d, "K/V dim mismatch");
    TORCH_CHECK(static_cast<int>(v.size(0)) == N, "V shape mismatch");

    auto out = torch::zeros({num_q, d}, q.options());
    int threads = 256;
    size_t smem_bytes = static_cast<size_t>(N + d) * sizeof(float);
    fused_mha_naive_kernel<<<num_q, threads, smem_bytes, get_stream()>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(), num_q, N, d, 1.0f / std::sqrt(static_cast<float>(d)));
    check_cuda(cudaGetLastError());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "GEMM (CUDA)");
    m.def("softmax", &softmax, "Softmax (CUDA)");
    m.def("layernorm", &layernorm, "LayerNorm (CUDA)");
    m.def("rmsnorm", &rmsnorm, "RMSNorm (CUDA)");
    m.def("block_reduce_sum", &block_reduce_sum_forward, "Block reduce sum (CUDA)");
    m.def("flash_attention", &flash_attention, "Flash attention (CUDA)");
    m.def("fused_mha", &fused_mha, "Fused MHA (CUDA)");
}
