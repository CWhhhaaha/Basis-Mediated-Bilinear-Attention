"""Shape-derived parameter and explicit-K/V cache accounting for integrations."""

from __future__ import annotations

import torch


def _positive(name: str, value: int):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def explicit_kv_cache_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    n_layers: int,
    n_kv_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Exact allocated payload size for contiguous explicit K and V tensors.

    This excludes allocator overhead, temporary tensors, model weights, and
    framework metadata. It counts both K and V across all requested layers.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("n_layers", n_layers),
        ("n_kv_heads", n_kv_heads),
        ("key_dim", key_dim),
        ("value_dim", value_dim),
    ):
        _positive(name, value)
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        batch_size
        * sequence_length
        * n_layers
        * n_kv_heads
        * (key_dim + value_dim)
        * element_size
    )


def compare_gqa_bmb_uv_cache(
    *,
    batch_size: int,
    sequence_length: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    factor_rank: int,
    value_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, int | float]:
    """Compare same-topology GQA and BMB-UV explicit-cache payload sizes."""
    common = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "value_dim": value_dim,
        "dtype": dtype,
    }
    gqa = explicit_kv_cache_bytes(key_dim=head_dim, **common)
    bmb_uv = explicit_kv_cache_bytes(key_dim=factor_rank, **common)
    return {
        "gqa_bytes": gqa,
        "bmb_uv_bytes": bmb_uv,
        "saved_bytes": gqa - bmb_uv,
        "reduction_fraction": 1.0 - bmb_uv / gqa,
        "gqa_gib": gqa / 2**30,
        "bmb_uv_gib": bmb_uv / 2**30,
    }


def compare_gqa_bmb_uv_qk_parameters(
    *,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    latent_rank: int,
    factor_rank: int,
) -> dict[str, int | float]:
    """Compare one layer's Q/K projection parameters, excluding V and output."""
    for name, value in (
        ("d_model", d_model),
        ("n_heads", n_heads),
        ("n_kv_heads", n_kv_heads),
        ("head_dim", head_dim),
        ("latent_rank", latent_rank),
        ("factor_rank", factor_rank),
    ):
        _positive(name, value)
    if n_heads % n_kv_heads:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    gqa = d_model * (n_heads + n_kv_heads) * head_dim
    bmb_uv = (
        d_model * latent_rank
        + (n_heads + n_kv_heads) * latent_rank * factor_rank
    )
    return {
        "gqa_qk_parameters": gqa,
        "bmb_uv_qk_parameters": bmb_uv,
        "saved_qk_parameters": gqa - bmb_uv,
        "qk_reduction_fraction": 1.0 - bmb_uv / gqa,
    }
