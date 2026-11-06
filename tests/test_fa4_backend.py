"""FA4 adapter contract tests using real PyTorch and a mocked FA4 entry point."""

import unittest
from unittest import mock

import torch

from bmb import BMBAttention, bmb_uv_attention, grouped_attention, grouped_sdpa
from bmb.attention import _fa4_attention, fa4_backend_info


class FakeTensor:
    ndim = 4
    is_cuda = True
    dtype = torch.bfloat16

    def __init__(self, shape):
        self.shape = tuple(shape)

    def transpose(self, dim0, dim1):
        shape = list(self.shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return FakeTensor(shape)


class FA4BackendTests(unittest.TestCase):
    def test_sdpa_does_not_import_fa4(self):
        q = torch.randn(1, 2, 3, 4)
        with mock.patch("bmb.attention.importlib.import_module") as loader:
            grouped_attention(q, q, q)
        loader.assert_not_called()

    def test_fa4_layout_causal_offset_and_native_gqa(self):
        q = FakeTensor((2, 4, 2, 32))
        k = FakeTensor((2, 2, 5, 32))
        v = FakeTensor((2, 2, 5, 64))
        calls = []

        def fake_fa4(q_bthd, k_bthd, v_bthd, **kwargs):
            calls.append((q_bthd, k_bthd, v_bthd, kwargs))
            return FakeTensor((2, 2, 4, 64)), None

        with mock.patch("bmb.attention._load_fa4", return_value=fake_fa4):
            output = _fa4_attention(
                q, k, v, is_causal=True, past_len=3, dropout_p=0.0
            )
        q_bthd, k_bthd, v_bthd, kwargs = calls[0]
        self.assertEqual(q_bthd.shape, (2, 2, 4, 32))
        self.assertEqual(k_bthd.shape, (2, 5, 2, 32))
        self.assertEqual(v_bthd.shape, (2, 5, 2, 64))
        self.assertEqual(kwargs["softmax_scale"], 32**-0.5)
        self.assertTrue(kwargs["causal"])
        self.assertEqual(output.shape, (2, 4, 2, 64))

    def test_missing_fa4_raises_install_error(self):
        q = FakeTensor((1, 2, 1, 32))
        k = FakeTensor((1, 1, 1, 32))
        v = FakeTensor((1, 1, 1, 64))
        with mock.patch(
            "bmb.attention.importlib.import_module",
            side_effect=ModuleNotFoundError("flash_attn"),
        ):
            with self.assertRaisesRegex(ImportError, "pip install flash-attn-4"):
                _fa4_attention(q, k, v, is_causal=True, past_len=0, dropout_p=0.0)

    def test_cpu_dropout_and_mask_are_rejected(self):
        q = torch.randn(1, 2, 1, 4)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            grouped_attention(q, q, q, backend="fa4")

        fake_q = FakeTensor((1, 2, 1, 32))
        fake_k = FakeTensor((1, 1, 1, 32))
        fake_v = FakeTensor((1, 1, 1, 64))
        with self.assertRaisesRegex(ValueError, "dropout"):
            grouped_attention(fake_q, fake_k, fake_v, dropout_p=0.1, backend="fa4")
        with self.assertRaisesRegex(ValueError, "padding or custom masks"):
            grouped_attention(
                fake_q, fake_k, fake_v, attention_mask=object(), backend="fa4"
            )

    def test_module_and_functional_dispatch_and_caches(self):
        module = BMBAttention(
            d_model=16,
            n_heads=4,
            n_kv_heads=2,
            latent_rank=8,
            factor_rank=4,
            value_dim=4,
            backend="fa4",
        )
        reference = BMBAttention.from_config(
            {**module.get_config(), "backend": "sdpa"}
        )
        reference.load_state_dict(module.state_dict())
        x = torch.randn(2, 5, 16)

        def sdpa_adapter(q, k, v, *, is_causal, past_len, dropout_p):
            return grouped_sdpa(
                q, k, v, is_causal=is_causal, past_len=past_len, dropout_p=dropout_p
            )

        with mock.patch("bmb.attention._fa4_attention", side_effect=sdpa_adapter) as fa4:
            actual, cache = module(x, use_cache=True)
            next_x = torch.randn(2, 1, 16)
            actual_next, _ = module(next_x, past_key_value=cache, use_cache=True)
            static = module.create_static_cache(2, 8)
            static_full, _ = module(x, past_key_value=static, use_cache=True)
            z = module.basis(x)
            value = module.v_proj(x).view(2, 5, 2, 4)
            functional, _ = bmb_uv_attention(
                z,
                module.q_factor,
                module.k_factor,
                value,
                cos_sin=module.rotary(x, 0, 5),
                backend="fa4",
            )
            projected = module.o_proj(functional.reshape(2, 5, 16))
        expected, expected_cache = reference(x, use_cache=True)
        expected_next, _ = reference(next_x, past_key_value=expected_cache)
        expected_functional, _ = bmb_uv_attention(
            reference.basis(x),
            reference.q_factor,
            reference.k_factor,
            reference.v_proj(x).view(2, 5, 2, 4),
            cos_sin=reference.rotary(x, 0, 5),
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_next, expected_next)
        torch.testing.assert_close(static_full, expected)
        torch.testing.assert_close(functional, expected_functional)
        torch.testing.assert_close(projected, expected)
        self.assertEqual(fa4.call_count, 4)

    def test_backend_info_reports_installed_distribution(self):
        with mock.patch("bmb.attention._load_fa4"), mock.patch(
            "bmb.attention.importlib.metadata.version", return_value="4.test"
        ):
            self.assertEqual(
                fa4_backend_info(), {"package": "flash-attn-4", "version": "4.test"}
            )

    def test_constructor_accepts_fa4_without_importing_it(self):
        with mock.patch("bmb.attention.importlib.import_module") as loader:
            module = BMBAttention(
                d_model=16, n_heads=4, latent_rank=8, factor_rank=4, backend="fa4"
            )
        self.assertEqual(module.backend, "fa4")
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
