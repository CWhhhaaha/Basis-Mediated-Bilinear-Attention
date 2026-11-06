"""Basis-Mediated Bilinear Attention."""

from bmb.benefits import (
    compare_gqa_bmb_uv_cache,
    compare_gqa_bmb_uv_qk_parameters,
    explicit_kv_cache_bytes,
)
from bmb.attention import (
    BMBAttention,
    RotaryEmbedding,
    apply_rope,
    bmb_uv_attention,
    grouped_attention,
    grouped_sdpa,
    project_bmb_uv,
)

__all__ = [
    "BMBAttention",
    "RotaryEmbedding",
    "apply_rope",
    "bmb_uv_attention",
    "grouped_attention",
    "grouped_sdpa",
    "project_bmb_uv",
    "compare_gqa_bmb_uv_cache",
    "compare_gqa_bmb_uv_qk_parameters",
    "explicit_kv_cache_bytes",
    "__version__",
]

__version__ = "1.2.0"
