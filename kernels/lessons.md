# 每日练习教训

## 通用

1. 对 shared mem 写入后需要做一次同步。
2. block reduce 第二阶段必须从 smem 读，不能复用寄存器旧值。warp 0 的线程在 `__syncthreads()` 后要先 `val = lid < nw ? smem[lid] : 0.0f` 再做 warp reduce，否则 val 还是本线程自己的旧值。
3. 同一翻译单元内不能有同名 `__device__` 函数。`bindings.cu` 把所有 `.cu` 用 `#include` 合并，各文件的辅助函数名要唯一（加文件前缀，如 `ln_`、`sm_`）。

## Flash Attention (Triton)

1. **Triton tensor shape 必须用 constexpr**：`tl.full` / `tl.zeros` 的 shape 只能包含 `tl.constexpr` 值。累加器 `o` 的第二维要写 `BLOCK_D`，而不是运行时变量 `d`。

2. **load mask 的广播方向要与指针算术一致**：加载形如 `[BLOCK_Q, BLOCK_D]` 的 tile 时，掩码应是 `offs_q[:, None] < N` & `offs_d[None, :] < d`（分别扩展成列向量和行向量）。把 `offs_d[:, None]` 写成列向量会让掩码 shape 变成 `(BLOCK_D, 1)` 而非 `(1, BLOCK_D)`，静默广播错误。K_T 的 bounds mask `offs_kv < N` 也要补 `[None, :]` 变成行向量，与 `(BLOCK_D, BLOCK_KV)` 对齐。

3. **online softmax rescale 符号**：旧输出的缩放因子是 `alpha = exp(m_old - m_new)`，不是 `exp(m_new - m_old)`。直觉：m_new ≥ m_old，所以 alpha ≤ 1，要把历史权重"缩小"再加新贡献。写反方向 rescale 会让分子指数爆炸。

4. **causal mask 需同时检查两个维度**：条件应为 `(offs_q[:, None] >= offs_kv[None, :]) & (offs_kv[None, :] < N)`。前者是因果条件，后者是 KV 越界保护——当 N 不被 BLOCK_KV 整除时，最后一个 tile 的越界 K 被填 0，若不 mask，`exp(0 - m_new)` 会错误地贡献到归一化分母。常见错误：只写 `offs_kv[:, None]`（变成列向量，广播成 `(BLOCK_Q, 1)` 而非 `(BLOCK_Q, BLOCK_KV)`，causal mask 完全退化），或只检查 Q 行的越界 `offs_q[:, None] < N` 而漏掉 KV 列。

5. **`tl.dot` 精度：用 `allow_tf32=False` 或 `input_precision="ieee"`**：A100 上 Triton 的 `tl.dot` 默认用 TF32（尾数 10 bit，float32 是 23 bit）。d=32 时累加误差约 32×2⁻¹⁰ ≈ 0.03，恰好压线；d=64/128 时误差翻倍，遇到参考值接近 0 的元素相对误差可达 10 倍以上。旧 API：`tl.dot(a, b, allow_tf32=False)`；新 API：`tl.dot(a, b, input_precision="ieee")`。`input_precision="tf32x3"` 是三轮近似，仍非精确 float32，不能替代 `"ieee"`。

6. **`m = m_new` 必须在 KV loop 末尾更新**：online softmax 依赖 `m` 记录所有已见 tile 的全局最大值。若循环体内只计算 `m_new` 而不执行 `m = m_new`，下一轮 `m` 仍是初始 `-inf`，导致 `alpha = exp(-inf - m_new) = 0`，前序累积输出全部清零，结果完全错误。这是 Flash Attention 最易漏写的一行。

7. **Triton grid launch 顺序与 `program_id` 一一对应**：`_kernel[grid]` 中 `grid = (dim0, dim1)` 对应 `program_id(0) ∈ [0, dim0)` 和 `program_id(1) ∈ [0, dim1)`。若内核用 `pid_q = program_id(0)`、`pid_bh = program_id(1)`，则启动时必须写 `grid = (cdiv(N, BLOCK_Q), B*H)`，而不是反过来。顺序写反后 Q-tile ID 和 batch/head ID 互换，整批计算都是错的，却不报任何错误。

8. **`tl.constexpr` 仅用于 kernel 内部**：在 host Python 中 `BLOCK_KV = tl.constexpr(32)` 是错误用法。`tl.constexpr` 是 `@triton.jit` 函数的参数类型标注，不是 Python 侧的构造函数。host 侧直接写 `BLOCK_KV = 32`，传参时 Triton 自动识别为 constexpr。

9. **`m_new` 广播方向**：`tl.exp(score - m_new)` 中 `score` 是 `[BLOCK_Q, BLOCK_KV]`，`m_new` 是 `[BLOCK_Q]`。按 numpy 规则，`[BLOCK_Q]` 会被视为 `[1, BLOCK_Q]` 广播，当 BLOCK_Q == BLOCK_KV 时不报错但结果错误（沿列减而非沿行减）。必须写 `tl.exp(score - m_new[:, None])` 显式扩展为列向量。

10. **完整的 Flash Attention 步骤清单（容易遗漏）**：
    - ① 计算分数时乘 scale（`* 1/sqrt(d)`）
    - ② 对每个 KV tile 应用 causal mask（同时检查因果条件和 KV 越界）
    - ③ `m_tile = max(S, axis=1)`；`m_new = maximum(m, m_tile)`；`alpha = exp(m - m_new)`
    - ④ `P = exp(S - m_new[:, None])`（注意 `[:, None]` 广播）
    - ⑤ `l = l * alpha + sum(P, axis=1)`；`O = O * alpha[:, None] + dot(P, V)`
    - ⑥ **`m = m_new`**（最易漏写）
    - ⑦ loop 结束后 `O = O / l[:, None]` 归一化
    - ⑧ `tl.store` 写回结果

## layernorm / rmsnorm

1. 计算均值时要用行指针 `rx[i]`，不能用全局指针 `x[i]`（`x[i]` 始终读第 0 行）。
2. 方差要除以 `cols`：block reduce 累加的是 `sum(d²)`，标准差是 `sqrt(sum(d²) / N)`，rmsnorm 的 rms 同理。
