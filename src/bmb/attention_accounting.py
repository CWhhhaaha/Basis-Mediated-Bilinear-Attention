from __future__ import annotations

from typing import Any


VARIANT_ALIASES = {
    "standard": "baseline",
    "layer_fully_shared": "fullyshared",
    "layer_partial_qk_shared": "partialshared",
    "layer_partial_shared": "headpartialshared",
    "layer_lowrank": "lowrank",
    "layer_bbt": "bbt",
    "layer_symmetric_latent": "bmb",
    "layer_uv_latent": "bmbuv",
}


def attention_forward_accounting(
    variant: str,
    *,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    tokens: int,
    rank: int | None = None,
    factor_rank: int | None = None,
    shared_dim: int | None = None,
) -> dict[str, Any]:
    """Weight counts and matrix-multiplication FLOPs for one forward pass.

    Biases, nonlinearities, masking, softmax, dropout, and reshapes are excluded.
    A multiply and an accumulate count as two FLOPs. FullShr applies its tied
    projection separately for Q and K. BB^T forms one score matrix per layer and
    broadcasts it across value heads.
    """
    variant = VARIANT_ALIASES.get(variant, variant)
    d = int(model_dim)
    h = int(num_heads)
    layers = int(num_layers)
    t = int(tokens)
    if d <= 0 or h <= 0 or layers <= 0 or t <= 0 or d % h:
        raise ValueError("Dimensions must be positive and model_dim must be divisible by num_heads")
    head_dim = d // h

    baseline_qk_params = 2 * d * d
    baseline_qk_flops = 4 * t * d * d + 2 * t * t * d
    value_output_params = 2 * d * d
    value_output_flops = 4 * t * d * d + 2 * t * t * d

    if variant == "baseline":
        qk_params = baseline_qk_params
        qk_flops = baseline_qk_flops
    elif variant == "fullyshared":
        qk_params = d * d
        qk_flops = baseline_qk_flops
    elif variant == "partialshared":
        if shared_dim is None or not 0 <= shared_dim <= d:
            raise ValueError("partialshared requires 0 <= shared_dim <= model_dim")
        private_dim = d - int(shared_dim)
        qk_params = d * int(shared_dim) + 2 * d * private_dim
        qk_flops = (
            2 * t * d * int(shared_dim)
            + 4 * t * d * private_dim
            + 2 * t * t * d
        )
    elif variant == "headpartialshared":
        if shared_dim is None or not 0 <= shared_dim <= head_dim:
            raise ValueError("headpartialshared requires 0 <= shared_dim <= head_dim")
        private_dim = head_dim - int(shared_dim)
        qk_params = d * int(shared_dim) + 2 * d * h * private_dim
        qk_flops = (
            2 * t * d * int(shared_dim)
            + 4 * t * d * h * private_dim
            + 2 * t * t * d
        )
    elif variant == "lowrank":
        if rank is None or rank <= 0:
            raise ValueError("lowrank requires a positive rank")
        qk_params = 2 * h * (d * int(rank) + int(rank) * head_dim)
        qk_flops = (
            4 * t * h * (d * int(rank) + int(rank) * head_dim)
            + 2 * t * t * d
        )
    elif variant == "bbt":
        if rank is None or rank <= 0:
            raise ValueError("bbt requires a positive rank")
        qk_params = d * int(rank)
        qk_flops = 2 * t * d * int(rank) + 2 * t * t * int(rank)
    elif variant == "bmb":
        if rank is None or rank <= 0:
            raise ValueError("bmb requires a positive rank")
        qk_params = d * int(rank) + (h + 1) * int(rank) ** 2
        qk_flops = (
            2 * t * d * int(rank)
            + 2 * h * t * int(rank) ** 2
            + 2 * h * t * t * int(rank)
        )
    elif variant == "bmbuv":
        if rank is None or rank <= 0:
            raise ValueError("bmbuv requires a positive rank")
        s = int(factor_rank if factor_rank is not None else rank)
        if s <= 0:
            raise ValueError("factor_rank must be positive")
        qk_params = d * int(rank) + 2 * h * int(rank) * s
        qk_flops = (
            2 * t * d * int(rank)
            + 4 * h * t * int(rank) * s
            + 2 * h * t * t * s
        )
    else:
        raise ValueError(f"Unknown attention variant: {variant}")

    attention_params = qk_params + value_output_params
    attention_flops = qk_flops + value_output_flops
    return {
        "variant": variant,
        "model_dim": d,
        "num_heads": h,
        "num_layers": layers,
        "head_dim": head_dim,
        "tokens": t,
        "qk_params_per_layer": int(qk_params),
        "attention_params_per_layer": int(attention_params),
        "qk_params_total": int(layers * qk_params),
        "attention_params_total": int(layers * attention_params),
        "qk_flops_per_layer": int(qk_flops),
        "attention_flops_per_layer": int(attention_flops),
        "qk_flops_total": int(layers * qk_flops),
        "attention_flops_total": int(layers * attention_flops),
        "baseline_qk_params_total": int(layers * baseline_qk_params),
        "baseline_attention_params_total": int(layers * (baseline_qk_params + value_output_params)),
        "baseline_qk_flops_total": int(layers * baseline_qk_flops),
        "baseline_attention_flops_total": int(layers * (baseline_qk_flops + value_output_flops)),
    }
