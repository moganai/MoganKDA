"""Chunked pure-torch forward / backward of the per-channel-gate (KDA) delta rule (no tilelang import).

Mirrors the kernel decomposition of the forward (`kkt_solve_kda` + `fused_fwd_kda`) and of the planned backward
(K0..K5) so that every kernel can be tested against the matching function:

    chunk_fwd_torch        forward (o, ht) + intermediates: g_c (log2 cumsum), A = Akk^-1, Aqk, q~, k~, kbar, gl, h, Vd
    scaled_operands_torch  K0: q~ = q 2^g, k~ = k 2^g, kbar = k 2^(g_L - g), gl = 2^g_L
    bwd_w_torch            K1: w = Akk diag(beta) k~
    bwd_dh_torch           K2: reverse chain -> dh_c = dS'_c (state gradient LEAVING chunk c), dVd, dh0, dgl
    bwd_post_torch         K3a/K3b: dV, dbeta_W, dAqk, dL, Vd, and the row-local partials dq_p, dk_p, dg'_p
    bwd_intra_torch        K4: dAqk / dL -> dq, dk, dg', dbeta contributions (strictly lower gate-carrying terms +
                           separate gate-free diagonal add: the diagonal never touches dg)
    bwd_final_torch        K5: reverse chunk-local cumsum of dg' and GVA sum of dq, dk
    chunk_bwd_torch        chains K0..K5

Conventions (identical to the kernels): q, k [B, T, HK, K]; v [B, T, HV, V]; g per-token natural-log decay
[B, T, HV, K]; beta [B, T, HV]; g_c = chunk-local cumsum of g in the log2 domain; A / Aqk / dAqk / dL are
[B, T, HV, 64] with row i = chunk-local row, column j; states [N, HV, K, V] ([N, HV, V, K] with state_v_first);
h / dh [lead, NC, HV, K, V] (state ENTERING / gradient LEAVING chunk c), gl / dgl [lead, NC, HV, K] with
lead = B, NC = ceil(T/64) (dense) or lead = 1, NC = total chunks (varlen, cu_seqlens; B must be 1). v-head h uses
q/k head h // (HV // HK). All math is in `dtype` (default float64); rows past a sequence end are absent (no padding).

`dg'` convention: gradient w.r.t. g_c as if d(2^x)/dx = 2^x (the ln2 / 1/ln2 factors cancel), so the gradient
w.r.t. the natural-log per-token g is a plain reverse chunk-local cumsum of dg'.

`anchored=True` reproduces the kernels' anchored factorisation of exp2(g_i - g_j) (i > j) as
exp2(g_i - g_a(i)) * exp2(g_a(i) - g_j) with a(i) = first row of i's 16-row sub-block (diagonal blocks: row 0 of the
sub-block, off-diagonal blocks: first row of the row block); mathematically identical, used for like-for-like numerics.
"""
import math

import torch

CHUNK = 64
SUB = 16
RCP_LN2 = 1.0 / math.log(2.0)


# ---------------------------------------------------------------------------
# chunk bookkeeping
# ---------------------------------------------------------------------------

def _chunk_list(B, T, cu_seqlens, C=CHUNK):
    """Returns (num_seqs, chunks, NC). Each chunk: n (sequence), b (batch row), s (start token), L (valid rows),
    ci / nc (index / count within its sequence), gb / gc (batch / chunk index into h, dh, gl, dgl)."""
    dense = cu_seqlens is None
    if dense:
        seqs = [(b, 0, T) for b in range(B)]
    else:
        assert B == 1, "cu_seqlens requires B == 1"
        cu = [int(x) for x in cu_seqlens.tolist()]
        seqs = [(0, cu[i], cu[i + 1]) for i in range(len(cu) - 1)]
    chunks, off = [], 0
    for n, (b, s, e) in enumerate(seqs):
        nc = -(-(e - s) // C)
        for ci in range(nc):
            st = s + ci * C
            chunks.append(dict(n=n, b=b, s=st, L=min(C, e - st), ci=ci, nc=nc,
                               gb=b if dense else 0, gc=ci if dense else off + ci))
        if not dense:
            off += nc
    NC = -(-T // C) if dense else off
    return len(seqs), chunks, NC


def _std(x, state_v_first):
    """state-like tensor from kernel layout to [.., K, V]."""
    return x.transpose(-1, -2) if state_v_first else x


def _pair_exp(g, anchored, strict):
    """E[i, j, h, c] = 2^(g_ic - g_jc) for i > j (strict) or i >= j, else 0. g [L, H, K]; masked BEFORE the exp."""
    L = g.shape[0]
    idx = torch.arange(L, device=g.device)
    mask = (idx[:, None] > idx[None, :]) if strict else (idx[:, None] >= idx[None, :])
    m = mask[:, :, None, None]
    if anchored:
        ga = g[(idx // SUB) * SUB]  # anchor row of every row i (first row of its sub-block, <= i)
        e1 = torch.exp2(g - ga)  # <= 1
        d2 = (ga[:, None] - g[None, :]).masked_fill(~m, 0)
        return (e1[:, None] * torch.exp2(d2)).masked_fill(~m, 0)
    d = (g[:, None] - g[None, :]).masked_fill(~m, 0)
    return torch.exp2(d).masked_fill(~m, 0)


def _z(dtype, device, *shape):
    return torch.zeros(*shape, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------

def scaled_operands_torch(q, k, g_c, cu_seqlens=None, chunk_size=CHUNK, dtype=torch.float64):
    """K0. Returns (qt, kt, kb, gl); qt/kt/kb [B, T, HV, K] (per v-head), gl [lead, NC, HV, K]."""
    q, k, g_c = q.to(dtype), k.to(dtype), g_c.to(dtype)
    B, T, HK, K = q.shape
    HV = g_c.shape[2]
    G = HV // HK
    qx, kx = q.repeat_interleave(G, 2), k.repeat_interleave(G, 2)
    _, chunks, NC = _chunk_list(B, T, cu_seqlens, chunk_size)
    lead = B if cu_seqlens is None else 1
    dev = q.device
    qt, kt, kb = (_z(dtype, dev, B, T, HV, K) for _ in range(3))
    gl = _z(dtype, dev, lead, NC, HV, K)
    for ch in chunks:
        b, sl, L = ch["b"], slice(ch["s"], ch["s"] + ch["L"]), ch["L"]
        gc = g_c[b, sl]
        Gm = torch.exp2(gc)
        gL = gc[L - 1]
        qt[b, sl] = qx[b, sl] * Gm
        kt[b, sl] = kx[b, sl] * Gm
        kb[b, sl] = kx[b, sl] * torch.exp2(gL[None] - gc)
        gl[ch["gb"], ch["gc"]] = torch.exp2(gL)
    return qt, kt, kb, gl


def chunk_fwd_torch(q, k, v, g, beta, scale=None, initial_state=None, cu_seqlens=None, state_v_first=False,
                    chunk_size=CHUNK, dtype=torch.float64, anchored=False):
    """Forward; returns a dict with o [B,T,HV,V], ht [N,HV,K,V] (kernel state layout) and the intermediates
    g_c, A, Aqk, qt, kt, kb, gl, h, Vd (see the module docstring)."""
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    B, T, HK, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    C = chunk_size
    scale = K ** -0.5 if scale is None else scale
    qx, kx = q.repeat_interleave(HV // HK, 2), k.repeat_interleave(HV // HK, 2)
    N, chunks, NC = _chunk_list(B, T, cu_seqlens, C)
    lead = B if cu_seqlens is None else 1
    dev = q.device
    z = lambda *s: _z(dtype, dev, *s)  # noqa: E731
    g_c, qt, kt, kb, o = z(B, T, HV, K), z(B, T, HV, K), z(B, T, HV, K), z(B, T, HV, K), z(B, T, HV, V)
    A, Aqk, Vd = z(B, T, HV, C), z(B, T, HV, C), z(B, T, HV, V)
    gl, h, ht = z(lead, NC, HV, K), z(lead, NC, HV, K, V), z(N, HV, K, V)
    h0 = None if initial_state is None else _std(initial_state.to(dtype), state_v_first)
    S = None
    for ch in chunks:
        b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
        if ch["ci"] == 0:
            S = z(HV, K, V) if h0 is None else h0[ch["n"]].clone()
        gc = torch.cumsum(g[b, sl], 0) * RCP_LN2  # [L, HV, K]
        g_c[b, sl] = gc
        kk, qq, bt = kx[b, sl], qx[b, sl], beta[b, sl]
        gam = torch.einsum("ihk,jhk,ijhk->hij", kk, kk, _pair_exp(gc, anchored, True))  # strictly lower
        Lm = gam * bt.transpose(0, 1)[:, :, None]
        eye = torch.eye(L, dtype=dtype, device=dev)
        Akk = torch.tril(torch.linalg.solve_triangular(eye + Lm, eye.expand(HV, L, L), upper=False))
        Aq = scale * torch.einsum("ihk,jhk,ijhk->hij", qq, kk, _pair_exp(gc, anchored, False))  # lower incl diag
        A[b, sl, :, :L] = Akk.permute(1, 0, 2)
        Aqk[b, sl, :, :L] = Aq.permute(1, 0, 2)
        Gm = torch.exp2(gc)
        gL = gc[L - 1]
        qt_, kt_, kb_ = qq * Gm, kk * Gm, kk * torch.exp2(gL[None] - gc)
        gl_ = torch.exp2(gL)
        qt[b, sl], kt[b, sl], kb[b, sl], gl[ch["gb"], ch["gc"]] = qt_, kt_, kb_, gl_
        h[ch["gb"], ch["gc"]] = S
        X = v[b, sl] - torch.einsum("lhk,hkv->lhv", kt_, S)
        W = bt[..., None] * X
        Vd_ = torch.einsum("hij,jhv->ihv", Akk, W)
        Vd[b, sl] = Vd_
        o[b, sl] = scale * torch.einsum("lhk,hkv->lhv", qt_, S) + torch.einsum("hij,jhv->ihv", Aq, Vd_)
        S = gl_[:, :, None] * S + torch.einsum("lhk,lhv->hkv", kb_, Vd_)
        if ch["ci"] == ch["nc"] - 1:
            ht[ch["n"]] = S
    if state_v_first:
        h, ht = h.transpose(-1, -2).contiguous(), ht.transpose(-1, -2).contiguous()
    return dict(o=o, ht=ht, g_c=g_c, A=A, Aqk=Aqk, qt=qt, kt=kt, kb=kb, gl=gl, h=h, Vd=Vd)


# ---------------------------------------------------------------------------
# backward kernels
# ---------------------------------------------------------------------------

def bwd_w_torch(A, beta, kt, cu_seqlens=None, chunk_size=CHUNK, dtype=torch.float64):
    """K1: w = Akk diag(beta) k~  [B, T, HV, K]."""
    A, beta, kt = A.to(dtype), beta.to(dtype), kt.to(dtype)
    B, T, HV, K = kt.shape
    _, chunks, _ = _chunk_list(B, T, cu_seqlens, chunk_size)
    w = _z(dtype, kt.device, B, T, HV, K)
    for ch in chunks:
        b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
        Akk = A[b, sl, :, :L].permute(1, 0, 2)
        w[b, sl] = torch.einsum("hij,jhk->ihk", Akk, beta[b, sl][..., None] * kt[b, sl])
    return w


def bwd_dh_torch(do, Aqk, w, qt, kb, gl, h, dht=None, scale=None, cu_seqlens=None, state_v_first=False,
                 chunk_size=CHUNK, dtype=torch.float64):
    """K2: reverse chain. For each chunk (last to first) with dS' = gradient of the state leaving the chunk:
         dh_c = dS';  dVd = Aqk^T dO + kbar dS';  dgl = rowsum_v(S (.) dS');
         dS = diag(gl) dS' + scale q~^T dO - w^T dVd        (S = h_c, the state entering the chunk)
    dS' of the last chunk of a sequence is dht (or 0); dh0 = dS of the first chunk.
    Returns dict(dh [lead,NC,HV,K,V], dVd [B,T,HV,V], dh0 [N,HV,K,V], dgl [lead,NC,HV,K])."""
    do, Aqk, w, qt, kb, gl, h = (x.to(dtype) for x in (do, Aqk, w, qt, kb, gl, h))
    B, T, HV, V = do.shape
    K = qt.shape[-1]
    scale = K ** -0.5 if scale is None else scale
    N, chunks, NC = _chunk_list(B, T, cu_seqlens, chunk_size)
    lead = B if cu_seqlens is None else 1
    dev = do.device
    dh, dgl = _z(dtype, dev, lead, NC, HV, K, V), _z(dtype, dev, lead, NC, HV, K)
    dVd, dh0 = _z(dtype, dev, B, T, HV, V), _z(dtype, dev, N, HV, K, V)
    h_std = _std(h, state_v_first)
    dht_std = None if dht is None else _std(dht.to(dtype), state_v_first)
    by_seq = [[] for _ in range(N)]
    for ch in chunks:
        by_seq[ch["n"]].append(ch)
    for n in range(N):
        dS = _z(dtype, dev, HV, K, V) if dht_std is None else dht_std[n].clone()
        for ch in reversed(by_seq[n]):
            b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
            gb, gc = ch["gb"], ch["gc"]
            dh[gb, gc] = dS
            dO = do[b, sl]
            Aq = Aqk[b, sl, :, :L].permute(1, 0, 2)
            dVd_ = torch.einsum("hij,ihv->jhv", Aq, dO) + torch.einsum("jhk,hkv->jhv", kb[b, sl], dS)
            dVd[b, sl] = dVd_
            dgl[gb, gc] = (h_std[gb, gc] * dS).sum(-1)
            dS = (gl[gb, gc][:, :, None] * dS + scale * torch.einsum("ihk,ihv->hkv", qt[b, sl], dO)
                  - torch.einsum("ihk,ihv->hkv", w[b, sl], dVd_))
        dh0[n] = dS
    if state_v_first:
        dh, dh0 = dh.transpose(-1, -2).contiguous(), dh0.transpose(-1, -2).contiguous()
    return dict(dh=dh, dVd=dVd, dh0=dh0, dgl=dgl)


def bwd_post_torch(do, v, h, dh, dVd, dgl, A, beta, qt, kt, kb, gl, g_c, scale=None, cu_seqlens=None,
                   state_v_first=False, chunk_size=CHUNK, dtype=torch.float64):
    """K3a + K3b (chunk-parallel). Recomputes X = V - k~ S, W = beta X, Vd = Akk W, then
      K3a: dAqk = tril(dO Vd^T) (lower incl diagonal, grad w.r.t. the stored Aqk), dW = Akk^T dVd, dX = beta dW,
           dV = dX, dbeta_W = rowsum_v(dW (.) X), dAkk = dVd W^T, dL = -StrictLower(Akk^T dAkk Akk^T);
      K3b: dq~ = scale dO S^T, dk~ = -dX S^T, dkbar = Vd dS'^T and the row-local gate terms
           dq_p = dq~ 2^g, dk_p = dk~ 2^g + dkbar 2^(g_L - g), dg'_p = q~ dq~ + k~ dk~ - kbar dkbar,
           dg'_p[L-1] += sum_j kbar_j dkbar_j + gl (.) dgl.
    Returns dict(dv, dbeta_w, dAqk, dL, Vd, dq_p, dk_p, dg_p)."""
    (do, v, h, dh, dVd, dgl, A, beta, qt, kt, kb, gl, g_c) = (
        x.to(dtype) for x in (do, v, h, dh, dVd, dgl, A, beta, qt, kt, kb, gl, g_c))
    B, T, HV, V = v.shape
    K = qt.shape[-1]
    C = chunk_size
    scale = K ** -0.5 if scale is None else scale
    _, chunks, _ = _chunk_list(B, T, cu_seqlens, C)
    dev = v.device
    z = lambda *s: _z(dtype, dev, *s)  # noqa: E731
    dv, Vd = z(B, T, HV, V), z(B, T, HV, V)
    dbeta_w = z(B, T, HV)
    dAqk, dL = z(B, T, HV, C), z(B, T, HV, C)
    dq_p, dk_p, dg_p = z(B, T, HV, K), z(B, T, HV, K), z(B, T, HV, K)
    h_std, dh_std = _std(h, state_v_first), _std(dh, state_v_first)
    for ch in chunks:
        b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
        gb, gc_i = ch["gb"], ch["gc"]
        S, dSp = h_std[gb, gc_i], dh_std[gb, gc_i]
        bt = beta[b, sl]
        Akk = A[b, sl, :, :L].permute(1, 0, 2)  # [HV, L, L]
        dO, dVd_ = do[b, sl], dVd[b, sl]
        X = v[b, sl] - torch.einsum("lhk,hkv->lhv", kt[b, sl], S)
        W = bt[..., None] * X
        Vd_ = torch.einsum("hij,jhv->ihv", Akk, W)
        # K3a
        dAq = torch.tril(torch.einsum("ihv,jhv->hij", dO, Vd_))
        dW = torch.einsum("hij,ihv->jhv", Akk, dVd_)
        dX = bt[..., None] * dW
        dAkk = torch.einsum("ihv,jhv->hij", dVd_, W)
        AkkT = Akk.transpose(-1, -2)
        dLm = -torch.tril(AkkT @ dAkk @ AkkT, -1)
        dv[b, sl], Vd[b, sl] = dX, Vd_
        dbeta_w[b, sl] = (dW * X).sum(-1)
        dAqk[b, sl, :, :L] = dAq.permute(1, 0, 2)
        dL[b, sl, :, :L] = dLm.permute(1, 0, 2)
        # K3b
        dqt = scale * torch.einsum("ihv,hkv->ihk", dO, S)
        dkt = -torch.einsum("ihv,hkv->ihk", dX, S)
        dkb = torch.einsum("jhv,hkv->jhk", Vd_, dSp)
        gcum = g_c[b, sl]
        Gm, gL = torch.exp2(gcum), gcum[L - 1]
        dq_p[b, sl] = dqt * Gm
        dk_p[b, sl] = dkt * Gm + dkb * torch.exp2(gL[None] - gcum)
        kbdkb = kb[b, sl] * dkb
        dgp = qt[b, sl] * dqt + kt[b, sl] * dkt - kbdkb
        dgp[L - 1] = dgp[L - 1] + kbdkb.sum(0) + gl[gb, gc_i] * dgl[gb, gc_i]
        dg_p[b, sl] = dgp
    return dict(dv=dv, dbeta_w=dbeta_w, dAqk=dAqk, dL=dL, Vd=Vd, dq_p=dq_p, dk_p=dk_p, dg_p=dg_p)


def bwd_intra_torch(q, k, g_c, beta, dAqk, dL, dq_p, dk_p, dg_p, scale=None, cu_seqlens=None,
                    chunk_size=CHUNK, dtype=torch.float64, anchored=False):
    """K4: intra-chunk gradients through Gamma / Aqk (per v-head). With E_ij = 2^(g_i - g_j), i > j:
        dq_i  += scale sum_{j<i} dAqk_ij k_j E_ij            dg'_i += q_i (that term)
        T_i    = sum_{j<i} dL_ij k_j E_ij ;  dk_i += beta_i T_i ;  dbeta_i = sum_c k_i T_i ;  dg'_i += k_i (beta_i T_i)
        dk_j  += sum_{i>j} (scale dAqk_ij q_i + beta_i dL_ij k_i) E_ij     dg'_j -= k_j (that term)
      diagonal, gate free:  dq_i += scale dAqk_ii k_i ,  dk_i += scale dAqk_ii q_i   (no dg contribution).
    All gate-carrying terms use the STRICTLY lower part of dAqk / dL, so the diagonal cannot reach dg (structural).
    Returns dict(dq, dk, dg (this is dg', partial dg_p included), dbeta_g); dq, dk are per v-head [B,T,HV,K]."""
    q, k, g_c, beta, dAqk, dL, dq_p, dk_p, dg_p = (x.to(dtype) for x in (q, k, g_c, beta, dAqk, dL, dq_p, dk_p, dg_p))
    B, T, HK, K = q.shape
    HV = g_c.shape[2]
    scale = K ** -0.5 if scale is None else scale
    qx, kx = q.repeat_interleave(HV // HK, 2), k.repeat_interleave(HV // HK, 2)
    _, chunks, _ = _chunk_list(B, T, cu_seqlens, chunk_size)
    dq, dk, dg = dq_p.clone(), dk_p.clone(), dg_p.clone()
    dbeta_g = _z(dtype, q.device, B, T, HV)
    for ch in chunks:
        b, L, sl = ch["b"], ch["L"], slice(ch["s"], ch["s"] + ch["L"])
        gc, qq, kk, bt = g_c[b, sl], qx[b, sl], kx[b, sl], beta[b, sl]
        dAq = dAqk[b, sl, :, :L].permute(1, 0, 2)  # [HV, L, L]
        dLm = dL[b, sl, :, :L].permute(1, 0, 2)
        idx = torch.arange(L, device=q.device)
        strict = idx[:, None] > idx[None, :]
        Wq = scale * dAq.masked_fill(~strict, 0)
        dLs = dLm.masked_fill(~strict, 0)
        dG = bt.transpose(0, 1)[:, :, None] * dLs  # beta_i dL_ij
        E = _pair_exp(gc, anchored, True)  # [L, L, HV, K]
        dq_row = torch.einsum("hij,jhk,ijhk->ihk", Wq, kk, E)
        T_ = torch.einsum("hij,jhk,ijhk->ihk", dLs, kk, E)
        dk_row = bt[..., None] * T_
        dk_col = (torch.einsum("hij,ihk,ijhk->jhk", Wq, qq, E) + torch.einsum("hij,ihk,ijhk->jhk", dG, kk, E))
        dg_int = qq * dq_row + kk * dk_row - kk * dk_col
        diag = torch.diagonal(dAq, dim1=-2, dim2=-1).transpose(0, 1)[..., None]  # [L, HV, 1]
        dq[b, sl] = dq[b, sl] + dq_row + scale * diag * kk
        dk[b, sl] = dk[b, sl] + dk_row + dk_col + scale * diag * qq
        dg[b, sl] = dg[b, sl] + dg_int
        dbeta_g[b, sl] = (kk * T_).sum(-1)
    return dict(dq=dq, dk=dk, dg=dg, dbeta_g=dbeta_g)


def bwd_final_torch(dg_prime, dq, dk, hk, cu_seqlens=None, chunk_size=CHUNK):
    """K5: dg = reverse chunk-local cumsum of dg' (natural-log per-token gradient, no scale); dq, dk summed over the
    GVA group [B,T,HV,K] -> [B,T,HK,K]."""
    B, T, HV, K = dq.shape
    _, chunks, _ = _chunk_list(B, T, cu_seqlens, chunk_size)
    dg = torch.zeros_like(dg_prime)
    for ch in chunks:
        b, sl = ch["b"], slice(ch["s"], ch["s"] + ch["L"])
        dg[b, sl] = dg_prime[b, sl].flip(0).cumsum(0).flip(0)
    G = HV // hk
    return dq.reshape(B, T, hk, G, K).sum(3), dk.reshape(B, T, hk, G, K).sum(3), dg


def chunk_bwd_torch(q, k, v, beta, g_c, A, Aqk, h, do, dht=None, scale=None, cu_seqlens=None,
                    state_v_first=False, chunk_size=CHUNK, dtype=torch.float64, anchored=False):
    """Chains K0..K5. Inputs are the forward's saved tensors (g_c, A, Aqk, h as produced by the kernels or by
    `chunk_fwd_torch`); returns dict(dq [B,T,HK,K], dk, dv [B,T,HV,V], dg [B,T,HV,K] w.r.t. the natural-log per-token
    input g, dbeta [B,T,HV], dh0 [N,HV,K,V] (kernel state layout)) plus the intermediates under "inter"."""
    q, k, v, beta, g_c, do = (x.to(dtype) for x in (q, k, v, beta, g_c, do))
    K = q.shape[-1]
    HK = q.shape[2]
    scale = K ** -0.5 if scale is None else scale
    kw = dict(cu_seqlens=cu_seqlens, chunk_size=chunk_size, dtype=dtype)
    qt, kt, kb, gl = scaled_operands_torch(q, k, g_c, **kw)
    w = bwd_w_torch(A, beta, kt, **kw)
    chain = bwd_dh_torch(do, Aqk, w, qt, kb, gl, h, dht, scale, state_v_first=state_v_first, **kw)
    post = bwd_post_torch(do, v, h, chain["dh"], chain["dVd"], chain["dgl"], A, beta, qt, kt, kb, gl, g_c, scale,
                          state_v_first=state_v_first, **kw)
    intra = bwd_intra_torch(q, k, g_c, beta, post["dAqk"], post["dL"], post["dq_p"], post["dk_p"], post["dg_p"],
                            scale, anchored=anchored, **kw)
    dq, dk, dg = bwd_final_torch(intra["dg"], intra["dq"], intra["dk"], HK, cu_seqlens, chunk_size)
    return dict(dq=dq, dk=dk, dv=post["dv"], dg=dg, dbeta=post["dbeta_w"] + intra["dbeta_g"], dh0=chain["dh0"],
                inter=dict(qt=qt, kt=kt, kb=kb, gl=gl, w=w, **chain, **post, intra=intra))
