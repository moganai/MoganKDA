"""KDA layer benchmark at large-model shapes (HK=16, HV=32, D=128; ``--small``: HK=8, HV=16) for packed-document workloads.

Per scenario: per-kernel fwd / bwd breakdown (CUDA events, median), totals, the whole-layer time through
``chunk_delta_rule`` (autograd, fwd+bwd), FLA ``chunk_kda`` fwd / bwd / total on the same inputs, and the parallel-chain
count of the state-chain kernels (fused_fwd: docs x HV x 2 V-halves; K2: docs x HV x 2 V-halves) against 132 SMs.

Option path: eda gate in kernel (raw pre-activation), sigmoid beta x2, L2 on q/k, silu epilogue.
FLA has no eda gate or epilogue: it runs the kimi gate in kernel (lower_bound -5), sigmoid beta and L2 natively (if this FLA
version supports them) and the epilogue in torch (excluded from FLA timings: reported as FLA core only). Anything FLA cannot
do is printed in the header.

Docs: lognormal (median 500, sigma 1.3 = heavy tail), seeded, cut at row boundaries; each row is its own set of segments,
rows concatenated as B = 1 varlen.

    python benchmark/bench_kda.py [--small] [--scen S1,S2,S3,S4,S5] [--iters 10]
"""
import argparse
import math
import os
import statistics
import sys
import traceback


import torch  # noqa: E402

from mogan_kda.ops.kimi_delta_rule.chunk import (  # noqa: E402
    chunk_delta_rule, chunk_kda_bwd, CHUNK_SIZE, RCP_LN2,
)
from mogan_kda.ops.kimi_delta_rule.chunk.hopper import (  # noqa: E402
    kkt_solve_kda, fused_fwd_kda, scaled_operands_kda, bwd_w_kda, bwd_dh_kda, bwd_post_kda, bwd_intra_kda,
)
from mogan_kda.ops.kimi_delta_rule.gate import kda_gate_cumsum_fwd, kda_gate_bwd  # noqa: E402
from mogan_kda.ops.kimi_delta_rule.epilogue import rmsnorm_gated_fwd, rmsnorm_gated_bwd  # noqa: E402
from mogan_kda.ops.utils import chunk_local_cumsum, group_reduce_vector  # noqa: E402
from mogan_kda.utils import l2norm_fwd, l2norm_bwd  # noqa: E402

DEV = "cuda"
SMS = 132
D = 128
LB = -5.0


# ---------------------------------------------------------------------------
# workload
# ---------------------------------------------------------------------------

def short_docs(gen, total, median=500, sigma=1.3, cap=4096):
    docs, s = [], 0
    while s < total:
        n = int(min(max(1, math.exp(math.log(median) + sigma * torch.randn(1, generator=gen).item())), cap))
        n = min(n, total - s)
        docs.append(n)
        s += n
    return docs


def row_docs(gen, row_len, long_share=0.0, long_range=(4096, 24576), fixed_long=None, single=False):
    if single:
        return [row_len]
    long_docs, used = [], 0
    if fixed_long:
        long_docs, used = [fixed_long], fixed_long
    while long_share and used < long_share * row_len:
        n = int(torch.randint(long_range[0], long_range[1] + 1, (1,), generator=gen).item())
        n = min(n, row_len - used)
        long_docs.append(n)
        used += n
    docs = long_docs + short_docs(gen, row_len - used)
    perm = torch.randperm(len(docs), generator=gen).tolist()
    return [docs[i] for i in perm]


def scenario_docs(name, seed=0):
    gen = torch.Generator().manual_seed(seed)
    if name == "S1":
        rows = [row_docs(gen, 8192) for _ in range(8)]
    elif name == "S2":
        rows = [row_docs(gen, 32768, long_share=0.2) for _ in range(2)]
    elif name == "S3":
        rows = [row_docs(gen, 65536, fixed_long=40960)]
    elif name == "S4":
        rows = [row_docs(gen, 65536, single=True)]
    elif name == "S5":
        rows = [row_docs(gen, 8192)]
    else:
        raise ValueError(name)
    docs = [d for r in rows for d in r]
    cu = [0]
    for d in docs:
        cu.append(cu[-1] + d)
    return docs, cu


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def med_us(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def make_inputs(Tn, HK, HV):
    torch.manual_seed(0)
    bf = torch.bfloat16
    return dict(
        q=torch.randn(1, Tn, HK, D, device=DEV, dtype=bf), k=torch.randn(1, Tn, HK, D, device=DEV, dtype=bf),
        v=torch.randn(1, Tn, HV, D, device=DEV, dtype=bf),
        x=torch.randn(1, Tn, HV, D, device=DEV), beta=torch.randn(1, Tn, HV, device=DEV),
        A_log=torch.empty(HV, device=DEV).uniform_(0.2, 4.0).log(), dt_bias=torch.randn(HV, D, device=DEV) * 0.5 - 1.0,
        ep_w=1.0 + 0.1 * torch.randn(D, device=DEV), ep_gate=torch.randn(1, Tn, HV, D, device=DEV, dtype=bf),
        do=torch.randn(1, Tn, HV, D, device=DEV),
    )


def breakdown(inp, cu, iters):
    """per-kernel medians (us) mirroring ChunkDeltaRuleKDAFunction forward / backward."""
    q, k, v, x, beta_raw = inp["q"], inp["k"], inp["v"], inp["x"], inp["beta"]
    A_log, dt_bias, ep_w, ep_gate, do = inp["A_log"], inp["dt_bias"], inp["ep_w"], inp["ep_gate"], inp["do"]
    scale = D ** -0.5
    F, B = {}, {}
    F["gate+cumsum"] = med_us(lambda: kda_gate_cumsum_fwd(x, "eda", A_log, dt_bias, LB, cu), iters)
    g_c = kda_gate_cumsum_fwd(x, "eda", A_log, dt_bias, LB, cu)
    beta = 2.0 * torch.sigmoid(beta_raw)
    F["beta(torch)"] = med_us(lambda: 2.0 * torch.sigmoid(beta_raw), iters)
    F["l2 q+k"] = med_us(lambda: (l2norm_fwd(q), l2norm_fwd(k)), iters)
    qn, q_rstd = l2norm_fwd(q)
    kn, k_rstd = l2norm_fwd(k)
    F["kkt(emit)"] = med_us(lambda: kkt_solve_kda(qn, kn, g_c, beta, scale=scale, chunk_size=CHUNK_SIZE, cu_seqlens=cu,
                                                   emit_scaled=True), iters)
    A, Aqk, qt, kt, kb, gl = kkt_solve_kda(qn, kn, g_c, beta, scale=scale, chunk_size=CHUNK_SIZE, cu_seqlens=cu,
                                           emit_scaled=True)
    fk = lambda: fused_fwd_kda(qt, kt, kb, v, A, Aqk, gl, beta, scale=scale, output_final_state=False, output_h=True,
                               cu_seqlens=cu)
    F["fused_fwd"] = med_us(fk, iters)
    o, h, _ = fk()
    F["epilogue"] = med_us(lambda: rmsnorm_gated_fwd(o, ep_w, ep_gate, "silu", 1e-6), iters)

    o_e = o
    B["epilogue bwd"] = med_us(lambda: rmsnorm_gated_bwd(do, o_e, ep_w, ep_gate, "silu", 1e-6), iters)
    dob = do.to(torch.bfloat16).contiguous()
    B["K0"] = med_us(lambda: scaled_operands_kda(qn, kn, g_c, cu), iters)
    qt, kt, kb, gl = scaled_operands_kda(qn, kn, g_c, cu)
    B["K1"] = med_us(lambda: bwd_w_kda(A, beta, kt, cu, negate=True), iters)
    nw = bwd_w_kda(A, beta, kt, cu, negate=True)
    B["K2"] = med_us(lambda: bwd_dh_kda(dob, Aqk, nw, qt, kb, gl, None, scale, cu, False, output_dh0=False), iters)
    dh, dvd, _ = bwd_dh_kda(dob, Aqk, nw, qt, kb, gl, None, scale, cu, False, output_dh0=False)
    kw = dict(q=qn, k=kn)
    B["K3a+K3b"] = med_us(lambda: bwd_post_kda(dob, v, h, dh, dvd, A, beta, qt, kt, kb, gl, g_c, scale, cu, False, **kw),
                          iters)
    post = bwd_post_kda(dob, v, h, dh, dvd, A, beta, qt, kt, kb, gl, g_c, scale, cu, False, **kw)
    k4 = lambda: bwd_intra_kda(qn, kn, g_c, beta, post["dAqk"], post["dL"], post["dq_p"], post["dk_p"], post["dg_p"],
                               scale, cu)
    B["K4"] = med_us(k4, iters)
    dq, dk, dgp, dbeta_g = k4()
    HK, HV = q.shape[2], v.shape[2]
    B["GVA reduce"] = med_us(lambda: (group_reduce_vector(dq, HK), group_reduce_vector(dk, HK)), iters) if HV > HK else 0.0
    B["gate bwd(+K5)"] = med_us(lambda: kda_gate_bwd(x, dgp, "eda", A_log, dt_bias, LB, cu, dg_is_prime=True,
                                                     zero_tail=True), iters)
    dq_r = group_reduce_vector(dq, HK) if HV > HK else dq
    B["l2 bwd"] = med_us(lambda: (l2norm_bwd(qn, q_rstd, dq_r), l2norm_bwd(kn, k_rstd, group_reduce_vector(dk, HK)
                                                                             if HV > HK else dk)), iters)
    return F, B


def whole_layer(inp, cu, iters):
    """chunk_delta_rule with the full option path, autograd fwd and fwd+bwd."""
    leaves = {n: inp[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "x", "beta", "A_log", "dt_bias",
                                                                     "ep_w", "ep_gate")}

    def fwd():
        return chunk_delta_rule(
            leaves["q"], leaves["k"], leaves["v"], leaves["x"], leaves["beta"], cu_seqlens=cu, gate_mode="eda",
            A_log=leaves["A_log"], dt_bias=leaves["dt_bias"], lower_bound=LB, beta_mode="sigmoid", beta_scale=2.0,
            use_qk_l2norm_in_kernel=True,
            output_norm=dict(weight=leaves["ep_w"], gate=leaves["ep_gate"], act="silu", eps=1e-6))[0]

    def fb():
        torch.autograd.grad((fwd() * inp["do"]).sum(), list(leaves.values()), allow_unused=True)

    with torch.no_grad():
        t_f = med_us(fwd, iters)
    return t_f, med_us(fb, iters)


def fla_times(inp, cu, iters):
    """FLA chunk_kda core (kimi gate in kernel, sigmoid beta / L2 native if available; no epilogue). Returns (fwd, bwd, total, notes)."""
    import inspect
    from fla.ops.kda import chunk_kda
    p = inspect.signature(chunk_kda).parameters
    kw = dict(use_gate_in_kernel=True, lower_bound=LB, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
    notes = ["kimi gate (no eda)", "GVA native" if True else ""]
    if "safe_gate" in p:
        kw["safe_gate"] = True
    if "use_beta_sigmoid_in_kernel" in p and "allow_neg_eigval" in p:
        kw.update(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    else:
        notes.append("beta sigmoid in torch")
    leaves = {n: inp[n].detach().clone().requires_grad_() for n in ("q", "k", "v", "x", "beta", "A_log", "dt_bias")}
    beta_fn = (lambda b: b) if "use_beta_sigmoid_in_kernel" in kw else (lambda b: 2.0 * torch.sigmoid(b))

    def run():
        return chunk_kda(leaves["q"], leaves["k"], leaves["v"], leaves["x"], beta_fn(leaves["beta"]),
                         A_log=leaves["A_log"], dt_bias=leaves["dt_bias"].reshape(-1), **kw)[0]

    with torch.no_grad():
        t_f = med_us(run, iters)
    ins = list(leaves.values())
    t_all = med_us(lambda: torch.autograd.grad((run() * inp["do"]).sum(), ins, allow_unused=True), iters)
    return t_f, t_all - t_f, t_all, notes


def chains(docs, HV):
    n_docs = len(docs)
    ctas = n_docs * HV * 2
    return n_docs, ctas


def run_scenario(name, HK, HV, iters):
    docs, cu_l = scenario_docs(name)
    Tn = cu_l[-1]
    cu = torch.tensor(cu_l, device=DEV, dtype=torch.int32)
    nchunks = sum(math.ceil(d / 64) for d in docs)
    print(f"\n=== {name}: T={Tn}, docs={len(docs)}, longest={max(docs)}, median={int(statistics.median(docs))}, "
          f"chunks={nchunks}, HK={HK} HV={HV} ===")
    nd, ctas = chains(docs, HV)
    print(f"state-chain kernels (fused_fwd, K2): chains = {nd} docs x {HV} HV x 2 V-halves = {ctas} CTAs "
          f"({ctas / SMS:.1f} waves of {SMS} SMs; longest chain {math.ceil(max(docs) / 64)} chunks, "
          f"chunk-steps/CTA-wave bound ~ {math.ceil(nchunks * HV * 2 / SMS)})")
    inp = make_inputs(Tn, HK, HV)
    F, Bk = breakdown(inp, cu, iters)
    tf, tfb = whole_layer(inp, cu, iters)
    print("fwd  (us): " + " | ".join(f"{n} {t:.0f}" for n, t in F.items()) + f" | sum {sum(F.values()):.0f}")
    print("bwd  (us): " + " | ".join(f"{n} {t:.0f}" for n, t in Bk.items()) + f" | sum {sum(Bk.values()):.0f}")
    print(f"layer via chunk_delta_rule: fwd {tf:.0f} | fwd+bwd {tfb:.0f} | bwd {tfb - tf:.0f} us")
    try:
        ff, fbw, ft, notes = fla_times(inp, cu, iters)
        print(f"FLA chunk_kda core ({'; '.join(notes)}; no epilogue): fwd {ff:.0f} | bwd {fbw:.0f} | total {ft:.0f} us"
              f"  -> ours/FLA total {tfb / ft:.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"FLA: {type(e).__name__}: {str(e)[:200]}")
    del inp
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="HK 8 / HV 16 (1B shape)")
    ap.add_argument("--scen", default="S1,S2,S3,S4,S5")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    assert a.iters >= 10
    HK, HV = (8, 16) if a.small else (16, 32)
    print(f"gate eda, beta sigmoid x2, L2, silu epilogue; GPU {torch.cuda.get_device_name()}")
    for s in a.scen.split(","):
        try:
            run_scenario(s.strip(), HK, HV, a.iters)
        except Exception as e:  # noqa: BLE001
            print(f"{s}: FAILED {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=3)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
