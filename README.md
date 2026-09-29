# MoganKDA

MoganKDA provides fused [TileLang](https://github.com/tile-ai/tilelang) kernels for **Kimi Delta Attention (KDA)**, the
delta rule with a per-channel decay gate, for NVIDIA Hopper (sm90). It covers the forward and backward pass for training
and inference, and moves the gate, beta, L2 normalisation and the output norm into the kernels. It is built on
[FlashQLA](https://github.com/QwenLM/FlashQLA), whose gated delta rule (GDN, scalar gate) kernels are included and
exposed through the same package.

## Features

- Per-channel gated delta rule (KDA), forward and backward, head dim 128.
- In-kernel options:
  - gate forms `eda`, `kimi`, `gdn` and `precomputed`, fused with the chunk-local cumsum, including their gradients
    (`A_log`, `dt_bias`);
  - sigmoid beta with a scale (`beta_scale * sigmoid(b)`);
  - q/k L2 normalisation;
  - a fused per-head gated RMSNorm output epilogue (`silu` or `sigmoid` gate).
- Grouped value heads (`HV` a multiple of `HK`), packed/varlen sequences through `cu_seqlens` (state reset per document),
  initial and final state, optional `[V, K]` state layout.
- Deterministic (no atomics) and **bitwise prefix-invariant**: outputs and gradients of the first P tokens do not change
  when later tokens change. The sub-block anchors used to keep `exp` in range are never later than the rows they serve,
  so no information leaks from the future; this makes packed training and cached prefixes reproducible.
- The backward keeps fp32-level accuracy for gradients that are sums with heavy cancellation (`A_log`, `dt_bias`) by
  splitting the fp32 operands of the relevant GEMMs into a high and a low bf16 part (hi/lo split).
- The gated delta rule (GDN) with its intra-card context parallelism is inherited from FlashQLA (`chunk_gated_delta_rule`).

## Installation

Requirements: an sm90 GPU (H100 / H200 / GH200) for the KDA kernels, CUDA 12.8 or newer, Python 3.10 or newer,
PyTorch 2.8 or newer and `tilelang==0.1.12`.

```bash
git clone https://github.com/moganai/MoganKDA.git
cd MoganKDA
pip install -e .
```

Kernels are JIT-compiled on first use (one compile per shape and option set, cached in `TILELANG_CACHE_DIR`).

## Usage

```python
import torch
from mogan_kda import chunk_kda

B, T, HK, HV, D = 1, 8192, 8, 16, 128
q = torch.randn(B, T, HK, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, HK, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, HV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
x = torch.randn(B, T, HV, D, device="cuda", requires_grad=True)          # raw gate pre-activation
b = torch.randn(B, T, HV, device="cuda", requires_grad=True)             # raw beta
A_log = torch.zeros(HV, device="cuda", requires_grad=True)
dt_bias = torch.zeros(HV, D, device="cuda", requires_grad=True)
cu_seqlens = torch.tensor([0, 3000, 5000, 8192], device="cuda", dtype=torch.int32)   # packed documents (B == 1)

o, final_state = chunk_kda(
    q, k, v, x, b,
    cu_seqlens=cu_seqlens,
    gate_mode="eda", A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,     # gate computed in the kernel
    beta_mode="sigmoid", beta_scale=2.0,
    use_qk_l2norm_in_kernel=True,
    output_norm=dict(weight=torch.ones(D, device="cuda"), gate=torch.randn(B, T, HV, D, device="cuda"), act="silu"),
)
o.sum().backward()
```

`chunk_kda(q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None,
state_v_first=False, gate_mode="precomputed", A_log=None, dt_bias=None, lower_bound=-5.0, beta_mode="precomputed",
beta_scale=1.0, use_qk_l2norm_in_kernel=False, output_norm=None)` returns `(o, final_state)`.

- `q`, `k`: `[B, T, HK, 128]` bf16; `v`: `[B, T, HV, 128]` bf16; `g`: `[B, T, HV, 128]` (fp32 or bf16); `beta`: `[B, T, HV]`.
- `cu_seqlens` (int32) requires `B == 1`. `initial_state`: fp32 `[N, HV, 128, 128]`.
- `gate_mode`: `"eda"` `g = l + (-l) exp(-(exp(A_log)/|l|) softplus(x + dt_bias))`; `"kimi"` `g = l sigmoid(exp(A_log)(x + dt_bias))`;
  `"gdn"` `g = max(-exp(A_log) softplus(x + dt_bias), l)` (the lower bound is enforced); `"precomputed"` `g = x`,
  the natural-log decay. `l = lower_bound`. `A_log`: `[HV]`, `dt_bias`: `[HV, 128]` or flat.
- `output_norm`: `dict(weight, gate, act, eps)`, `RMSNorm(o) * weight * act(gate)` per head; the output is then fp32.
- `chunk_delta_rule` is the same function and also accepts a scalar gate (`g` of shape `[B, T, HV]`, GDN kernels).
  `chunk_gated_delta_rule` is the GDN entry point inherited from FlashQLA.

## Benchmarks

Single H-class Hopper GPU (GH200), one KDA layer, forward + backward, HK = 16, HV = 32, D = 128, 64k tokens per GPU,
packed documents (lognormal lengths, median 500), bf16. `benchmark/bench_kda.py`.

| scenario | MoganKDA (ms) | FLA `chunk_kda` (ms) |
|---|---|---|
| S1: 8 rows x 8k, short documents | 31.2 | 37.9 |
| S2: 2 rows x 32k, short documents + 20% long documents (4k-24k) | 30.7 | 38.6 |
| S3: 1 row x 64k, short documents + one 40k document | 30.7 | 39.5 |
| S4: one 64k document | 30.4 | 40.9 |
| S5: 1 row x 8k | 4.6 | 5.4 |

MoganKDA runs the full option path (eda gate, sigmoid beta, L2, fused epilogue). FLA has no eda gate or output epilogue,
so its timing covers `chunk_kda` only, without the epilogue; the comparison favours FLA. At the 1B shape (HK = 8, HV = 16,
`--small`) the MoganKDA / FLA time ratio is 0.66-0.79.

Training throughput, 1B hybrid model (18 KDA layers, HK = 8, HV = 16), same data and seed. Baseline: FLA `chunk_kda` with the
gate, L2 normalisation and output norm computed in torch. MoganKDA: gate, beta, L2 and epilogue in the kernels.

| setup | FLA | MoganKDA |
|---|---|---|
| 1 node x 4 GH200 | 29.2k tok/s/GPU | 34.1k tok/s/GPU (+17%) |
| 8 nodes | 26.4k tok/s/GPU | 31.2k tok/s/GPU (+18%) |
| peak memory (1 node) | 80.9 GB | 74.2 GB |

The loss curves of the two backends agree to four decimals.

## Numerical notes

- **Gate contract.** The per-token natural-log decay must satisfy `g >= -5`. The kernels evaluate `exp2` of differences
  of chunk-local cumulative gates inside 16-row sub-blocks, and this bound keeps every exponent inside the fp32 range
  (`15 * 5 / ln 2 < 127`). `eda`, `kimi` and `gdn` (with the enforced lower bound) satisfy it by construction;
  with `precomputed` gates it is the caller's responsibility.
- **Prefix invariance.** Anchoring a sub-block at its middle row (as `flash-linear-attention` did before
  [fla#1240](https://github.com/fla-org/flash-linear-attention/issues/1240)) lets outputs of early tokens depend on later
  gates. MoganKDA anchors every sub-block at its first row, so tokens depend only on earlier tokens, and results are
  bitwise identical when the future changes.
- **Accuracy.** Against an fp64 reference, forward outputs agree to a max relative error below 2e-2 and every gradient
  below 5e-2 (typically 1e-2 or better). In the hardest measured case (gdn gate, gradient of `A_log`, a sum over all tokens
  and channels) the error is 2.2e-2, compared with 2.4e-2 for FLA. `dq` and `dk` leave the backward as bf16.

## Testing

A GPU is required for most tests (the CPU-only reference tests run anywhere).

```bash
pip install -e ".[test]"
pytest tests -m "not gpu"                          # fp64 reference / oracle tests, no GPU
KDA_IMPL=qla,fla pytest tests/test_kda_unit.py     # option matrix vs fp64 reference and FLA (sm90)
pytest tests/test_kda_options.py tests/test_kda_gate.py
pytest tests/test_kkt_solve_kda.py tests/test_fused_fwd_kda.py tests/test_bwd_*_kda.py tests/test_kda_bwd_kernels.py
python benchmark/bench_kda.py --scen S1,S4        # layer benchmark (add --small for HK 8 / HV 16)
```

`KDA_FULL=1` runs the full option matrix in `test_kda_unit.py`. The GDN tests inherited from FlashQLA are
`tests/test_gdr_unit.py` and `tests/test_cp_*.py`.

## Scope

MoganKDA targets Hopper (sm90) for KDA. The inherited FlashQLA context parallelism (CP) applies to GDN; KDA context
parallelism is not included.

## Acknowledgements and license

MoganKDA is derived from [FlashQLA](https://github.com/QwenLM/FlashQLA) (Qwen team, Alibaba, MIT), and its algorithms and
test methodology follow [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT); FLA is also the
reference implementation used in the benchmarks. Released under the MIT License (see `LICENSE`): Copyright (c) 2026 Qwen,
Alibaba; Copyright (c) 2026 MoganAI.
