# 每日练习教训

## 通用

1. 对 shared mem 写入后需要做一次同步。
2. block reduce 第二阶段必须从 smem 读，不能复用寄存器旧值。warp 0 的线程在 `__syncthreads()` 后要先 `val = lid < nw ? smem[lid] : 0.0f` 再做 warp reduce，否则 val 还是本线程自己的旧值。
3. 同一翻译单元内不能有同名 `__device__` 函数。`bindings.cu` 把所有 `.cu` 用 `#include` 合并，各文件的辅助函数名要唯一（加文件前缀，如 `ln_`、`sm_`）。

## layernorm / rmsnorm

1. 计算均值时要用行指针 `rx[i]`，不能用全局指针 `x[i]`（`x[i]` 始终读第 0 行）。
2. 方差要除以 `cols`：block reduce 累加的是 `sum(d²)`，标准差是 `sqrt(sum(d²) / N)`，rmsnorm 的 rms 同理。
