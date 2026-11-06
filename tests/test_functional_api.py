import io
import unittest

import torch

from bmb import (
    BMBAttention,
    bmb_uv_attention,
    compare_gqa_bmb_uv_cache,
    compare_gqa_bmb_uv_qk_parameters,
)


class FunctionalBMBUVTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.attention = BMBAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            latent_rank=16,
            factor_rank=8,
            value_dim=12,
        ).double().eval()
        self.x = torch.randn(2, 7, 32, dtype=torch.float64)

    def assertClose(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    def functional_inputs(self, x):
        value = self.attention.v_proj(x).view(2, x.shape[1], 2, 12)
        return self.attention.basis(x), value

    def test_functional_operator_matches_trainable_module(self):
        z, value = self.functional_inputs(self.x)
        cos_sin = self.attention.rotary(self.x, 0, self.x.shape[1])
        with torch.no_grad():
            functional, _ = bmb_uv_attention(
                z,
                self.attention.q_factor,
                self.attention.k_factor,
                value,
                cos_sin=cos_sin,
            )
            functional = self.attention.o_proj(functional.reshape(2, 7, 48))
            module, _ = self.attention(self.x)
        self.assertClose(functional, module)

    def test_functional_dynamic_cache_matches_full_sequence(self):
        z, value = self.functional_inputs(self.x)
        with torch.no_grad():
            expected, _ = bmb_uv_attention(
                z, self.attention.q_factor, self.attention.k_factor, value
            )
            cache = None
            chunks = []
            for z_chunk, v_chunk in (
                (z[:, :3], value[:, :3]),
                (z[:, 3:6], value[:, 3:6]),
                (z[:, 6:], value[:, 6:]),
            ):
                output, cache = bmb_uv_attention(
                    z_chunk,
                    self.attention.q_factor,
                    self.attention.k_factor,
                    v_chunk,
                    past_key_value=cache,
                    use_cache=True,
                )
                chunks.append(output)
        self.assertClose(torch.cat(chunks, dim=1), expected)

    def test_functional_static_cache_and_bidirectional_attention(self):
        z, value = self.functional_inputs(self.x)
        static = self.attention.create_static_cache(2, 8, dtype=torch.float64)
        with torch.no_grad():
            expected, _ = bmb_uv_attention(
                z, self.attention.q_factor, self.attention.k_factor, value
            )
            first, static = bmb_uv_attention(
                z[:, :4], self.attention.q_factor, self.attention.k_factor,
                value[:, :4], past_key_value=static, use_cache=True,
            )
            second, static = bmb_uv_attention(
                z[:, 4:], self.attention.q_factor, self.attention.k_factor,
                value[:, 4:], past_key_value=static, use_cache=True,
            )
            self.assertClose(torch.cat((first, second), dim=1), expected)

            sdpa, _ = bmb_uv_attention(
                z, self.attention.q_factor, self.attention.k_factor, value,
                is_causal=False, backend="sdpa",
            )
            reference, _ = bmb_uv_attention(
                z, self.attention.q_factor, self.attention.k_factor, value,
                is_causal=False, backend="reference",
            )
        self.assertClose(reference, sdpa)
        self.assertEqual(static["length"], 7)

    def test_reference_backend_matches_sdpa_with_padding(self):
        reference = BMBAttention.from_config({
            **self.attention.get_config(), "backend": "reference"
        }).double().eval()
        reference.load_state_dict(self.attention.state_dict())
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
        positions = torch.tensor([
            [0, 0, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 5],
        ])
        with torch.no_grad():
            expected, _ = self.attention(
                self.x, attention_mask=mask, position_ids=positions
            )
            actual, _ = reference(
                self.x, attention_mask=mask, position_ids=positions
            )
        self.assertClose(actual, expected)

    def test_mha_mqa_and_gqa_head_configurations(self):
        for kv_heads in (4, 1, 2):
            with self.subTest(n_kv_heads=kv_heads):
                attention = BMBAttention(
                    d_model=32,
                    n_heads=4,
                    n_kv_heads=kv_heads,
                    latent_rank=16,
                    factor_rank=8,
                    value_dim=12,
                ).double().eval()
                with torch.no_grad():
                    output, cache = attention(self.x, use_cache=True)
                self.assertEqual(output.shape, self.x.shape)
                self.assertEqual(cache[0].shape, (2, kv_heads, 7, 8))

    def test_optional_and_external_rope(self):
        without_rope = BMBAttention(
            d_model=32,
            n_heads=4,
            n_kv_heads=2,
            latent_rank=16,
            factor_rank=8,
            value_dim=12,
            rope=False,
        ).double().eval()
        without_rope.load_state_dict(self.attention.state_dict())
        cos_sin = self.attention.rotary(self.x, 0, self.x.shape[1])
        with torch.no_grad():
            external, _ = without_rope(self.x, cos_sin=cos_sin)
            internal, _ = self.attention(self.x)
        self.assertClose(external, internal)

        odd_width = BMBAttention(
            d_model=32,
            n_heads=4,
            latent_rank=16,
            factor_rank=7,
            value_dim=12,
            rope=False,
        )
        self.assertEqual(odd_width(self.x.float())[0].shape, self.x.shape)

    def test_config_and_state_dict_round_trip(self):
        restored = BMBAttention.from_config(self.attention.get_config()).double().eval()
        buffer = io.BytesIO()
        torch.save(self.attention.state_dict(), buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            expected, _ = self.attention(self.x)
            actual, _ = restored(self.x)
        self.assertClose(actual, expected)

    def test_functional_backward_reaches_all_inputs(self):
        z = torch.randn(1, 5, 4, dtype=torch.float64, requires_grad=True)
        q_factor = torch.randn(4, 4, 6, dtype=torch.float64, requires_grad=True)
        k_factor = torch.randn(2, 4, 6, dtype=torch.float64, requires_grad=True)
        value = torch.randn(1, 5, 2, 3, dtype=torch.float64, requires_grad=True)
        output, _ = bmb_uv_attention(z, q_factor, k_factor, value)
        output.square().mean().backward()
        for tensor in (z, q_factor, k_factor, value):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_dynamo_eager_wrapper_preserves_functional_output(self):
        z, value = self.functional_inputs(self.x)
        compiled = torch.compile(bmb_uv_attention, backend="eager")
        with torch.no_grad():
            expected, _ = bmb_uv_attention(
                z, self.attention.q_factor, self.attention.k_factor, value
            )
            actual, _ = compiled(
                z, self.attention.q_factor, self.attention.k_factor, value
            )
        self.assertClose(actual, expected)

    def test_xl_cache_and_qk_accounting(self):
        cache = compare_gqa_bmb_uv_cache(
            batch_size=1,
            sequence_length=1_000_000,
            n_layers=48,
            n_kv_heads=2,
            head_dim=64,
            factor_rank=32,
            value_dim=64,
        )
        self.assertEqual(cache["reduction_fraction"], 0.25)
        self.assertEqual(cache["bmb_uv_bytes"], 18_432_000_000)
        self.assertEqual(cache["gqa_bytes"], 24_576_000_000)

        parameters = compare_gqa_bmb_uv_qk_parameters(
            d_model=1600,
            n_heads=48,
            n_kv_heads=2,
            head_dim=64,
            latent_rank=128,
            factor_rank=32,
        )
        self.assertEqual(parameters["gqa_qk_parameters"], 5_120_000)
        self.assertEqual(parameters["bmb_uv_qk_parameters"], 409_600)


if __name__ == "__main__":
    unittest.main()
