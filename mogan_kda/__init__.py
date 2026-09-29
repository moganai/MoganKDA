# Copyright (c) 2026 MoganAI.
# Derived from FlashQLA (Copyright (c) 2026 The Qwen team, Alibaba Group). Licensed under The MIT License [see LICENSE].

__version__ = "0.1.0"

from mogan_kda.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule,
)
from mogan_kda.ops.kimi_delta_rule import chunk_kda, chunk_delta_rule

__all__ = [
    "chunk_kda",
    "chunk_delta_rule",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
]
