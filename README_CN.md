# CUDA 算子练习 &nbsp;·&nbsp; [English](README.md)

一个从零实现并基准测试 LLM 推理 GPU 算子的结构化练习环境。

涵盖六个核心算子，每个算子都有参考实现、练习骨架文件，以及自动验证正确性、对比 PyTorch 速度的测试框架。

---

## 算子列表

| 算子 | 签名 | 说明 |
| --- | --- | --- |
| **GEMM** | `C[M,N] = A[M,K] @ B[K,N]` | 两级分块：64×64 块分块 + 4×4 寄存器分块 |
| **Softmax** | 逐行，数值稳定 | 两趟：先求 max，再 exp/sum |
| **LayerNorm** | `(x - μ) / σ * γ + β` | 两趟：均值 + 方差 |
| **RMSNorm** | `x / rms(x) * γ` | 单趟，LLaMA 中使用 |
| **Block Reduce** | 对任意数组求和或取最大值 | 两 kernel 模式：分块 partial → final |
| **Flash Attention** | 单头，支持因果 mask 或全注意力 | 分块 K/V 的 online softmax |
| **Fused MHA** | 多头 decode / prefill | 融合 QKV 投影 + 注意力计算 |

---

## 环境依赖

- CUDA toolkit ≥ 11.8（nvcc、cuda_runtime.h）
- Python 3.10+
- [uv](https://docs.astral.sh/uv/)（推荐）或 pip

---

## 快速开始

### 1. 克隆并初始化环境

```bash
git clone <repo-url>
cd cudaEnv
uv sync
source .venv/bin/activate
```

`uv sync` 会读取 [pyproject.toml](pyproject.toml)，自动安装 PyTorch（CUDA 12.8 wheel）和 Triton 到 `.venv`。

如需切换 CUDA 版本，在 `pyproject.toml` 中修改 index URL（如 `cu121`、`cu124`）后重新运行 `uv sync`。

### 2. 创建一个练习 session

```bash
python kernels/new_session.py --out kernels/sessions/$(date +%Y-%m-%d)
```

生成六个算子的骨架文件：

```text
kernels/sessions/2026-04-02/
├── gemm.cu                # CUDA 骨架 — 在这里实现
├── softmax.cu
├── layernorm.cu
├── block_reduce.cu
├── flash_attention.cu
├── fused_mha.cu
├── gemm_triton.py         # Triton 骨架 — 或者在这里实现
├── softmax_triton.py
├── ...
└── bindings.cu            # 自动生成的 PyTorch C++ 绑定，不需要手动修改
```

### 3. 实现 kernel

打开 `gemm.cu`（或任意算子文件），里面有 kernel 函数和 `launch_*` 函数的桩代码：

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

填写 kernel 逻辑。`bindings.cu` 会按照 `launch_*` 的命名约定调用这些函数，保持函数签名不变即可。

也可以改为实现对应的 `*_triton.py`。

### 4. 验证正确性

```bash
# 验证单个算子
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --op gemm --backend cuda --mode verify

# 一次验证所有算子
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --backend cuda --mode verify
```

输出对每个测试 shape 显示 PASS / FAIL / ERR：

```text
gemm  [64, 64, 64]        PASS
gemm  [128, 256, 512]     PASS
gemm  [1000, 1000, 1000]  FAIL  max_err=0.012
```

### 5. 基准测试

```bash
python kernels/run.py \
    --session kernels/sessions/2026-04-02 \
    --op gemm --backend cuda --mode bench
```

在 LLM 常见规模的 shape 上与 torch（cuBLAS）对比：

```text
gemm  [4096, 4096, 4096]   cuda 3.21 ms   torch 1.05 ms   0.33x
gemm  [4096, 11008, 4096]  cuda 8.42 ms   torch 2.81 ms   0.33x
```

Triton backend 用法完全相同，替换 `--backend triton` 即可。

---

## Backend 说明

| Backend | 运行的代码 | 用途 |
| --- | --- | --- |
| `torch` | PyTorch 原生算子（cuBLAS 等） | 参考实现 / 速度上限 |
| `std` | `kernels/reference/` 参考 CUDA 实现 | 查看可运行的参考答案 |
| `cuda` | 你的 session `*.cu` 文件 | 你的 CUDA 实现 |
| `triton` | 你的 session `*_triton.py` 文件 | 你的 Triton 实现 |

不带 `--session` 运行 `--backend torch --mode bench` 可以获得当前 GPU 的参考性能基线。

---

## 项目结构

```text
cudaEnv/
├── kernels/
│   ├── run.py                    # 验证 / 基准测试主入口
│   ├── new_session.py            # session 生成器
│   ├── llm_kernels.cu            # 六个算子合并在一个文件（注释详细）
│   ├── reference/                # 按算子拆分的参考实现
│   │   ├── common.cuh            # warp/block reduction 公共工具
│   │   ├── gemm.cu
│   │   ├── softmax.cu
│   │   ├── layernorm.cu
│   │   ├── block_reduce.cu
│   │   ├── flash_attention.cu
│   │   ├── fused_mha.cu
│   │   └── bindings.cu
│   ├── sessions/
│   │   └── YYYY-MM-DD/           # 每天一个练习文件夹
│   │       ├── *.cu
│   │       ├── *_triton.py
│   │       └── bindings.cu
│   └── lessons.md                # 开发过程中踩过的坑
├── examples/
│   ├── gemm/                     # GEMM 分步优化演示
│   │   ├── 01_gemm_native.cu     # 朴素基线（每线程负责一个输出元素）
│   │   ├── 02_gemm_tile.cu       # shared memory 分块
│   │   ├── 03_gemm_reg.cu        # 寄存器分块
│   │   └── benchmark.py
│   └── flash_attention/          # Flash Attention 独立实验
│       ├── naive_attention.py
│       ├── flash_attention.py
│       └── benchmark.py
├── pyproject.toml
├── README.md
└── README_CN.md
```

### 推荐阅读顺序

1. **`kernels/llm_kernels.cu`** — 从这里开始。六个算子集中在一个文件中，注释详细解释了分块策略、warp/block reduction 模式以及 online softmax 的数学推导。
2. **`kernels/reference/`** — 相同实现按算子拆分成独立文件，阅读完整体版本后可以按需查阅单个算子。
3. **`examples/gemm/`** — 如果对 GEMM 分块不熟悉，`01_gemm_native.cu` 展示了引入 shared memory 和寄存器分块之前最简单的写法。

---

## run.py 参数说明

```text
python kernels/run.py [选项]

  --session PATH      session 文件夹路径（使用 cuda/triton backend 时必须提供）
  --op OP             要测试的算子：gemm | softmax | layernorm | block_reduce
                      | flash_attention | fused_mha  （默认：全部）
  --variant VARIANT   rmsnorm（layernorm 算子的变体）、decode | prefill（fused_mha 变体）
  --backend BACKEND   torch | std | cuda | triton  （默认：所有可用 backend）
  --mode MODE         verify | bench  （默认：verify）
```

---

## 常见踩坑

实现这些 kernel 时反复遇到的问题：

- **shared memory 写入后必须 `__syncthreads()`**，再由其他线程读取，否则读到的是旧数据。
- **block reduce 第二阶段**：`__syncthreads()` 之后要重新从 shared memory 读值到寄存器，不能复用第一阶段的寄存器旧值。
- **`bindings.cu` 把所有 `.cu` 用 `#include` 合并到同一翻译单元**，各算子的 `__device__` 辅助函数名必须唯一，建议加文件前缀（`ln_`、`sm_`、`fa_` 等）。
- **LayerNorm 均值计算**：用行指针 `rx[i]` 索引，不要用全局指针 `x[i]`，否则每行都读第 0 行的数据。
- **方差分母**：除以 `cols`（元素个数），不是 block reduce 累加的原始值。

---

## License

MIT
