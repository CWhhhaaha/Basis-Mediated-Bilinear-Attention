"""Calculate XL parameter and explicit-cache savings from model shapes."""

from bmb import (
    compare_gqa_bmb_uv_cache,
    compare_gqa_bmb_uv_qk_parameters,
)


cache = compare_gqa_bmb_uv_cache(
    batch_size=1,
    sequence_length=1_000_000,
    n_layers=48,
    n_kv_heads=2,
    head_dim=64,
    factor_rank=32,
    value_dim=64,
)
parameters = compare_gqa_bmb_uv_qk_parameters(
    d_model=1600,
    n_heads=48,
    n_kv_heads=2,
    head_dim=64,
    latent_rank=128,
    factor_rank=32,
)

print("XL, batch 1, 1M positions, BF16 explicit K/V payload")
print(f"  GQA:     {cache['gqa_gib']:.2f} GiB")
print(f"  BMB-UV:  {cache['bmb_uv_gib']:.2f} GiB")
print(f"  reduction: {100 * cache['reduction_fraction']:.2f}%")
print("One-layer Q/K projection parameters (excluding V and output)")
print(f"  GQA:     {parameters['gqa_qk_parameters']:,}")
print(f"  BMB-UV:  {parameters['bmb_uv_qk_parameters']:,}")
print(f"  reduction: {100 * parameters['qk_reduction_fraction']:.2f}%")
