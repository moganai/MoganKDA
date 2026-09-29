# Per-channel-gate (KDA) delta rule kernels. Derived from mogan_kda.ops.gated_delta_rule (Qwen team, MIT).

from .chunk import chunk_kda, chunk_kda_fwd, chunk_kda_bwd, chunk_delta_rule, kkt_solve_kda, fused_fwd_kda


__all__ = ["chunk_kda", "chunk_kda_fwd", "chunk_kda_bwd", "chunk_delta_rule", "kkt_solve_kda", "fused_fwd_kda"]
