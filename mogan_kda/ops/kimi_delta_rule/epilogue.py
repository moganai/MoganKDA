"""Output epilogue of the KDA layer: per-head RMSNorm(o) * weight * act(gate), forward and backward, TileLang.

Semantics (fla ``FusedRMSNormGated`` / ``chunk._output_norm``): with n = o * rsqrt(mean_d(o^2) + eps),
    y = n * w * act(gate),   act = silu ("swish") | sigmoid,   o [B, T, HV, D], gate [B, T, HV, D], w [D] (shared
over heads, optional). y is fp32.

Kernels: rows = B*T*HV vectors of length D (power of 2, 128 in practice); grid = ceil(rows / 16) programs of 128
threads, one program = 16 rows. Forward: reduce sum(o^2) over D (``T.reduce_sum`` dim=1), y = o * rs * w * act(gate).
Backward (rstd is recomputed from o, nothing saved): with a = act(gate), dn = dy * w * a,
    do = rs * (dn - n * mean_d(dn * n)),   dgate = dy * n * w * act'(gate),   dw = sum_rows dy * n * a.
dw is deterministic without atomics: each program reduces its 16 rows (``T.reduce_sum`` dim=0) into a partial row
``part_w[program, :]`` and dw is the fixed-shape torch sum of the partials over programs. Padded tail rows are loaded
as zeros (dy = o = 0) and contribute exactly 0.
"""
import torch
import tilelang
import tilelang.language as T

from .gate import _sigmoid

BLOCK_R = 16


def _norm_act(act):
    assert act in ("silu", "swish", "sigmoid"), f"unknown epilogue act {act!r}"
    return "sigmoid" if act == "sigmoid" else "silu"


def _act(act, g):
    return g * _sigmoid(g) if act == "silu" else _sigmoid(g)


def _act_grad(act, g):
    sg = _sigmoid(g)
    return sg * (1.0 + g * (1.0 - sg)) if act == "silu" else sg * (1.0 - sg)


@tilelang.jit
def tilelang_rmsnorm_gated_fwd(D, act, eps, o_dtype, g_dtype, block_R=BLOCK_R):
    N = T.dynamic("N")

    @T.prim_func
    def rmsnorm_gated_fwd_kernel(
        o: T.Tensor([N, D], dtype=o_dtype),
        gate: T.Tensor([N, D], dtype=g_dtype),
        w: T.Tensor([D], dtype="float32"),
        y: T.Tensor([N, D], dtype="float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_R), threads=128) as (bx,):
            base = bx * block_R
            o_f = T.alloc_fragment((block_R, D), dtype=o_dtype)
            g_f = T.alloc_fragment((block_R, D), dtype=g_dtype)
            sq_f = T.alloc_fragment((block_R, D), dtype="float32")
            y_f = T.alloc_fragment((block_R, D), dtype="float32")
            ss = T.alloc_fragment((block_R,), dtype="float32")

            if base + block_R <= N:
                T.copy(o[base:base + block_R, :], o_f)
                T.copy(gate[base:base + block_R, :], g_f)
            else:
                for r, d in T.Parallel(block_R, D):
                    if base + r < N:
                        o_f[r, d] = o[base + r, d]
                        g_f[r, d] = gate[base + r, d]
                    else:
                        o_f[r, d] = 0
                        g_f[r, d] = 0

            for r, d in T.Parallel(block_R, D):
                sq_f[r, d] = T.Cast("float32", o_f[r, d]) * T.Cast("float32", o_f[r, d])
            T.reduce_sum(sq_f, ss, dim=1, clear=True)

            for r, d in T.Parallel(block_R, D):
                y_f[r, d] = (
                    T.Cast("float32", o_f[r, d]) * T.rsqrt(ss[r] * (1.0 / D) + eps) * w[d]
                    * _act(act, T.Cast("float32", g_f[r, d]))
                )

            if base + block_R <= N:
                T.copy(y_f, y[base:base + block_R, :])
            else:
                for r, d in T.Parallel(block_R, D):
                    if base + r < N:
                        y[base + r, d] = y_f[r, d]

    return rmsnorm_gated_fwd_kernel


@tilelang.jit
def tilelang_rmsnorm_gated_bwd(D, act, eps, o_dtype, g_dtype, block_R=BLOCK_R):
    N = T.dynamic("N")
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def rmsnorm_gated_bwd_kernel(
        dy: T.Tensor([N, D], dtype="float32"),
        o: T.Tensor([N, D], dtype=o_dtype),
        gate: T.Tensor([N, D], dtype=g_dtype),
        w: T.Tensor([D], dtype="float32"),
        do: T.Tensor([N, D], dtype=o_dtype),
        dgate: T.Tensor([N, D], dtype=g_dtype),
        part_w: T.Tensor([num_blocks, D], dtype="float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_R), threads=128) as (bx,):
            base = bx * block_R
            dy_f = T.alloc_fragment((block_R, D), dtype="float32")
            o_f = T.alloc_fragment((block_R, D), dtype=o_dtype)
            g_f = T.alloc_fragment((block_R, D), dtype=g_dtype)
            n_f = T.alloc_fragment((block_R, D), dtype="float32")
            dn_f = T.alloc_fragment((block_R, D), dtype="float32")
            tmp_f = T.alloc_fragment((block_R, D), dtype="float32")
            do_f = T.alloc_fragment((block_R, D), dtype=o_dtype)
            dg_f = T.alloc_fragment((block_R, D), dtype=g_dtype)
            ss = T.alloc_fragment((block_R,), dtype="float32")
            dot = T.alloc_fragment((block_R,), dtype="float32")
            dw_red = T.alloc_fragment((D,), dtype="float32")

            if base + block_R <= N:
                T.copy(dy[base:base + block_R, :], dy_f)
                T.copy(o[base:base + block_R, :], o_f)
                T.copy(gate[base:base + block_R, :], g_f)
            else:
                for r, d in T.Parallel(block_R, D):
                    if base + r < N:
                        dy_f[r, d] = dy[base + r, d]
                        o_f[r, d] = o[base + r, d]
                        g_f[r, d] = gate[base + r, d]
                    else:
                        dy_f[r, d] = 0
                        o_f[r, d] = 0
                        g_f[r, d] = 0

            for r, d in T.Parallel(block_R, D):
                tmp_f[r, d] = T.Cast("float32", o_f[r, d]) * T.Cast("float32", o_f[r, d])
            T.reduce_sum(tmp_f, ss, dim=1, clear=True)

            # n = o * rs, dn = dy * w * act(gate)
            for r, d in T.Parallel(block_R, D):
                n_f[r, d] = T.Cast("float32", o_f[r, d]) * T.rsqrt(ss[r] * (1.0 / D) + eps)
                dn_f[r, d] = dy_f[r, d] * w[d] * _act(act, T.Cast("float32", g_f[r, d]))
            for r, d in T.Parallel(block_R, D):
                tmp_f[r, d] = dn_f[r, d] * n_f[r, d]
            T.reduce_sum(tmp_f, dot, dim=1, clear=True)

            for r, d in T.Parallel(block_R, D):
                do_f[r, d] = T.rsqrt(ss[r] * (1.0 / D) + eps) * (dn_f[r, d] - n_f[r, d] * dot[r] * (1.0 / D))
                dg_f[r, d] = dy_f[r, d] * n_f[r, d] * w[d] * _act_grad(act, T.Cast("float32", g_f[r, d]))

            # dw partial: sum over the rows of the program of dy * n * act(gate)
            for r, d in T.Parallel(block_R, D):
                tmp_f[r, d] = dy_f[r, d] * n_f[r, d] * _act(act, T.Cast("float32", g_f[r, d]))
            T.reduce_sum(tmp_f, dw_red, dim=0, clear=True)
            for d in T.Parallel(D):
                part_w[bx, d] = dw_red[d]

            if base + block_R <= N:
                T.copy(do_f, do[base:base + block_R, :])
                T.copy(dg_f, dgate[base:base + block_R, :])
            else:
                for r, d in T.Parallel(block_R, D):
                    if base + r < N:
                        do[base + r, d] = do_f[r, d]
                        dgate[base + r, d] = dg_f[r, d]

    return rmsnorm_gated_bwd_kernel


def _weight(w, o):
    D = o.shape[-1]
    if w is None:
        return torch.ones(D, device=o.device, dtype=torch.float32)
    assert w.numel() == D, "epilogue weight must be [D] (shared over heads)"
    return w.float().reshape(D).contiguous()


def _check(o, gate):
    D = o.shape[-1]
    assert D & (D - 1) == 0, "epilogue head dim must be a power of 2"
    assert o.is_contiguous() and gate.is_contiguous() and gate.shape == o.shape
    assert o.dtype in (torch.float32, torch.bfloat16) and gate.dtype in (torch.float32, torch.bfloat16)


def rmsnorm_gated_fwd(o, weight, gate, act="silu", eps=1e-6):
    """y (fp32, o.shape) = RMSNorm(o) * weight * act(gate)."""
    _check(o, gate)
    D = o.shape[-1]
    y = torch.empty(o.shape, device=o.device, dtype=torch.float32)
    kernel = tilelang_rmsnorm_gated_fwd(D, _norm_act(act), float(eps), o.dtype, gate.dtype)
    kernel(o.view(-1, D), gate.view(-1, D), _weight(weight, o), y.view(-1, D))
    return y


def rmsnorm_gated_bwd(dy, o, weight, gate, act="silu", eps=1e-6):
    """(do [o.dtype], dw fp32 [D] | None, dgate [gate.dtype]); deterministic (per-program partials + torch sum)."""
    _check(o, gate)
    D = o.shape[-1]
    dy = dy.float().contiguous()
    do, dgate = torch.empty_like(o), torch.empty_like(gate)
    N = o.numel() // D
    part_w = torch.empty(tilelang.cdiv(N, BLOCK_R), D, device=o.device, dtype=torch.float32)
    kernel = tilelang_rmsnorm_gated_bwd(D, _norm_act(act), float(eps), o.dtype, gate.dtype)
    kernel(dy.view(-1, D), o.view(-1, D), gate.view(-1, D), _weight(weight, o), do.view(-1, D), dgate.view(-1, D),
           part_w)
    dw = None if weight is None else part_w.sum(0).reshape(weight.shape).to(weight.dtype)
    return do, dw, dgate


class RMSNormGatedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, weight, gate, act, eps):
        o, gate = o.contiguous(), gate.contiguous()
        ctx.save_for_backward(o, weight, gate)
        ctx.act, ctx.eps = act, eps
        return rmsnorm_gated_fwd(o, weight, gate, act, eps)

    @staticmethod
    def backward(ctx, dy):
        o, weight, gate = ctx.saved_tensors
        do, dw, dgate = rmsnorm_gated_bwd(dy, o, weight, gate, ctx.act, ctx.eps)
        return do, dw, dgate, None, None


def rmsnorm_gated(o, weight, gate, act="silu", eps=1e-6):
    """Autograd-enabled per-head gated RMSNorm (fp32 output)."""
    return RMSNormGatedFunction.apply(o, weight, gate, act, eps)


__all__ = ["rmsnorm_gated_fwd", "rmsnorm_gated_bwd", "rmsnorm_gated"]
