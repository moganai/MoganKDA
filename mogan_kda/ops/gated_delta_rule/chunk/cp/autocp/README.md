# FlashQLA Auto CP

[English](#english) · [中文](#中文)

---

## English

**Auto CP** decides *whether* to enable intra-card CP and, if so, *what subsequence length* to use. It predicts the latency of a small set of candidate configurations with a lightweight kernel model and compares the best CP candidate against no-CP.

Auto CP has three steps:

1. **Kernel latency model** — predict each kernel's latency from its launch shape.
2. **CP partition search** — evaluate a small set of subsequence lengths near execution-wave boundaries.
3. **CP enablement** — enable CP only if the best CP candidate improves over no-CP by a sufficient margin.

### 1. Kernel latency model

Every GDN kernel repeats a fixed-granularity unit of work, such as a chunk-level recurrence step or a CP-partition operation. We abstract the workload of a CTA by its work-unit count and fit one linear model per kernel row:

$$
t_\kappa = a \cdot d_\kappa + b \cdot u + c,
$$

where:

* $d_\kappa$ — **critical schedule depth**: the number of work units on the busiest SM under a greedy CTA-to-SM assignment. It is computed by `schedule_depth()` in `utils.py`, which greedily packs CTA runs onto the available SMs and returns the resulting makespan.
* $u = H \cdot n_{\mathrm{part}}$ — **CTA population**: the total number of CTAs, capturing the additional launch and scheduling cost introduced by a larger CTA population.
* $a, b, c$ — fitted coefficients for each kernel family and hardware architecture.

With every kernel modeled, FlashQLA Auto CP supports two optimization modes selected by `is_train`. In `infer` mode, Auto CP minimizes the summed latency of all CP-related kernels in the forward pass. In `train` mode, it minimizes the summed latency of all CP-related kernels across both the forward and backward passes.

Fitted coefficients are stored in `coefs/<arch>.csv`; the mapping from each kernel's `block_DV` to its corresponding coefficient row is resolved in `launch.py`.

### 2. CP partition search

For a given workload, Auto CP seeks the subsequence length $L$ that minimizes the predicted CP latency. Exhaustively evaluating every supported length, however, would introduce unnecessary online search overhead. Auto CP therefore focuses on candidates near **execution-wave boundaries**, where GPU utilization can change sharply.

For a batch of $B$ sequences with lengths $S_b$, $H$ independent heads, and $P$ SMs, the recurrent kernel assigns one CTA to each partition-head pair. A subsequence length that approximately fills $w$ execution waves is

$$
L_{\mathrm{wave}}^{(w)} \approx \frac{H \sum_{b=1}^{B} S_b}{wP}.
$$

Intuitively, this chooses a partition length that produces approximately $wP$ CTAs, filling $w$ execution waves.

`wave_candidates()` enumerates wave counts $w$ in ascending order. For each wave boundary, it computes $L_{\mathrm{wave}}^{(w)}$ and maps it to nearby supported subsequence lengths. Candidate generation stops once the predefined candidate budget `MAX_EVALS` is reached.

The latency model then evaluates this bounded candidate set and selects

$$
L_{\mathrm{CP}}^{*} = \underset{L \in \mathcal{L}_{\mathrm{wave}}}{\arg\min}\; \widehat{T}\!\left(x(L)\right),
$$

where $x(L)$ denotes the CP configuration induced by subsequence length $L$.

This wave-aware search concentrates evaluations around configurations that are most likely to change GPU occupancy, while keeping the online decision overhead small.

### 3. CP enablement

Enabling CP introduces additional partitioning and correction overhead, so CP is unnecessary when the original workload already exposes sufficient parallelism.

Auto CP compares the predicted latency of the best CP configuration against the no-CP configuration and enables CP only if

$$
T_{\mathrm{cp}} < T_{\mathrm{nocp}} \cdot (1-\mathrm{margin}),
$$

where `margin` defaults to 0.05.

### Usage

```python
from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp import decide

use_cp, lcp = decide(
    num_chunks=[...],      # or cu_seqlens=... / seq_lens=...
    num_v_heads=H,
    P=sm_count,            # SM count used by the wave-aware partition search
    chunk=chunk_size,
    is_train=False,
    g=gate_tensor,         # optional; enables warmup-aware prepare_h depth
)

# use_cp: bool
# lcp: max local chunks (None when CP is disabled)
```

Auto CP is integrated into FlashQLA as the default partitioning strategy for intra-card CP.

### Calibration

Coefficients for supported architectures are shipped in `coefs/<arch>.csv`. To re-fit the latency model or adapt Auto CP to a new architecture, run `scripts/fit_autocp.sh`. The calibration implementation is located in `tools/autocp/`.

---

## 中文

**Auto CP** 用于决定*是否*开启卡内 CP，以及开启时采用*多长的子序列*。它使用轻量的 kernel 时间模型预测少量候选配置的延迟，并将最优 CP 配置与不开启 CP 的配置进行比较。

Auto CP 包含三个步骤：

1. **kernel 时间建模** —— 根据 launch shape 预测各个 kernel 的执行时间。
2. **CP 切分搜索** —— 在 execution-wave 边界附近评估少量候选子序列长度。
3. **CP 开启决策** —— 仅当最优 CP 配置相比不开 CP 有足够收益时才开启 CP。

### 1. kernel 时间建模

GDN kernel 都会重复执行某种固定粒度的工作单元，例如逐 chunk 的递推计算或逐 CP partition 的操作。我们使用 CTA 承担的工作单元数抽象其负载，并为每个 kernel row 拟合一个线性模型：

$$
t_\kappa = a \cdot d_\kappa + b \cdot u + c,
$$

其中：

* $d_\kappa$ —— **关键调度深度（critical schedule depth）**：将 CTA 贪心分配到各个 SM 后，最忙 SM 上累计的工作单元数。该值由 `utils.py` 中的 `schedule_depth()` 计算，其通过将 CTA runs 贪心打包到可用 SM 上，并返回最终的 makespan。
* $u = H \cdot n_{\mathrm{part}}$ —— **CTA 数量（CTA population）**：总 CTA 数，用于刻画 CTA 数量增加带来的额外 launch 和调度开销。
* $a,b,c$ —— 针对不同 kernel family 和硬件架构分别拟合得到的系数。

完成各个 kernel 的建模后，FlashQLA Auto CP 支持由 `is_train` 控制的两种优化模式。`infer` 模式优化前向过程中所有 CP 相关 kernel 的总执行时间；`train` 模式则优化前向和反向过程中所有 CP 相关 kernel 的总执行时间，分别对应推理和训练场景。

拟合得到的系数保存在 `coefs/<arch>.csv` 中。不同 kernel 的 `block_DV` 与具体 coefficient row 之间的映射由 `launch.py` 负责。

### 2. CP 切分搜索

对于给定 workload，Auto CP 希望找到使预测 CP 延迟最小的子序列长度 $L$。如果在线枚举所有支持的长度，会引入不必要的搜索开销。因此 Auto CP 将搜索集中在 **execution-wave 边界**附近，因为 GPU 利用率通常会在这些位置发生明显变化。

对于一个包含 $B$ 条序列的 batch，其序列长度分别为 $S_b$，共有 $H$ 个独立 head 和 $P$ 个 SM。递推 kernel 为每一个 partition-head pair 分配一个 CTA。若希望 workload 大约填满 $w$ 个 execution wave，则对应的子序列长度可以近似写为

$$
L_{\mathrm{wave}}^{(w)} \approx \frac{H \sum_{b=1}^{B} S_b}{wP}.
$$

直观上，这一长度会产生大约 $wP$ 个 CTA，从而恰好填充约 $w$ 个 execution wave。

`wave_candidates()` 按照 wave 数 $w$ 从小到大进行枚举。对于每个 wave 边界，首先计算对应的 $L_{\mathrm{wave}}^{(w)}$，再将其映射到附近实际支持的子序列长度。当候选数量达到预设上限 `MAX_EVALS` 后停止继续生成候选。

随后，latency model 在这一有限候选集合中选择预测时间最短的配置：

$$
L_{\mathrm{CP}}^{*} = \underset{L \in \mathcal{L}_{\mathrm{wave}}}{\arg\min}\; \widehat{T}\!\left(x(L)\right),
$$

其中 $x(L)$ 表示由子序列长度 $L$ 对应的 CP 配置。

这种 wave-aware 的搜索方式将有限的在线评估集中在最可能改变 GPU occupancy 的位置，同时控制 Auto CP 本身的决策开销。

### 3. CP 开启决策

开启 CP 会额外引入 partition 和 correction 等开销，因此当原始 workload 已经具有足够并行度时，并不一定需要开启 CP。

Auto CP 将最优 CP 配置的预测执行时间与不开启 CP 的配置进行比较，仅当

$$
T_{\mathrm{cp}} < T_{\mathrm{nocp}} \cdot (1-\mathrm{margin})
$$

时开启 CP，其中 `margin` 默认值为 0.05。

### 用法

```python
from mogan_kda.ops.gated_delta_rule.chunk.cp.autocp import decide

use_cp, lcp = decide(
    num_chunks=[...],      # 或 cu_seqlens=... / seq_lens=...
    num_v_heads=H,
    P=sm_count,            # wave-aware partition search 使用的 SM 数
    chunk=chunk_size,
    is_train=False,
    g=gate_tensor,         # 可选；启用 warmup-aware prepare_h depth
)

# use_cp: bool
# lcp: max local chunks（不开启 CP 时为 None）
```

Auto CP 已集成到 FlashQLA 中，作为 intracard CP 默认的 partitioning strategy。

### 标定系数

支持架构对应的拟合系数保存在 `coefs/<arch>.csv` 中。如果需要重新拟合 latency model，或者适配新的硬件架构，可以运行 `scripts/fit_autocp.sh`。相关标定实现位于 `tools/autocp/`。