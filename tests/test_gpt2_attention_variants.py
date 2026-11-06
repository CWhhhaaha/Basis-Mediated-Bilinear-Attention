from types import SimpleNamespace

import pytest
import torch

from bmb.gpt2_attention_variants import (
    GPT2BBTAttention,
    GPT2FullySharedAttention,
    GPT2LowRankAttention,
    GPT2PartialSharedAttention,
    GPT2UVLatentAttention,
)


def _config():
    return SimpleNamespace(
        n_head=2,
        n_embd=8,
        n_layer=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        initializer_range=0.02,
    )


def _padding_mask():
    keep = torch.tensor(
        [[True, True, True, True], [True, True, False, False]]
    )
    additive = torch.zeros(2, 1, 1, 4)
    return keep, additive.masked_fill(~keep[:, None, None, :], float("-inf"))


@pytest.mark.parametrize(
    "variant",
    [
        lambda config: GPT2LowRankAttention(config, low_rank=3),
        GPT2FullySharedAttention,
        lambda config: GPT2UVLatentAttention(
            config, latent_rank=6, latent_factor_rank=4
        ),
        lambda config: GPT2PartialSharedAttention(config, shared_qk_dim=6),
    ],
    ids=["lowrank", "fullyshared", "bmbuv", "partialshared"],
)
@pytest.mark.parametrize("mask_type", ["additive", "boolean"])
def test_sdpa_with_hf_padding_mask_matches_manual_attention(variant, mask_type):
    torch.manual_seed(7)
    module = variant(_config()).eval()
    hidden_states = torch.randn(2, 4, 8)
    keep, additive_mask = _padding_mask()
    attention_mask = (
        additive_mask
        if mask_type == "additive"
        else keep[:, None, None, :].expand(-1, -1, 4, -1)
    )

    with torch.no_grad():
        sdpa_output, _, _ = module(
            hidden_states, attention_mask=attention_mask, output_attentions=False
        )
        manual_output, _, weights = module(
            hidden_states, attention_mask=attention_mask, output_attentions=True
        )

    torch.testing.assert_close(sdpa_output, manual_output, atol=1e-6, rtol=1e-6)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    allowed = causal[None, None] & keep[:, None, None, :]
    assert torch.count_nonzero(weights.masked_select(~allowed.expand_as(weights))) == 0


@pytest.mark.parametrize(
    "variant",
    [
        lambda config: GPT2LowRankAttention(config, low_rank=3),
        GPT2FullySharedAttention,
        lambda config: GPT2UVLatentAttention(
            config, latent_rank=6, latent_factor_rank=4
        ),
        lambda config: GPT2PartialSharedAttention(config, shared_qk_dim=6),
    ],
    ids=["lowrank", "fullyshared", "bmbuv", "partialshared"],
)
def test_sdpa_without_mask_keeps_builtin_causal_path(variant):
    torch.manual_seed(11)
    module = variant(_config()).eval()
    hidden_states = torch.randn(2, 4, 8)

    with torch.no_grad():
        sdpa_output, _, _ = module(hidden_states, output_attentions=False)
        manual_output, _, weights = module(hidden_states, output_attentions=True)

    torch.testing.assert_close(sdpa_output, manual_output, atol=1e-6, rtol=1e-6)
    future = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights.masked_select(future[None, None])) == 0


def test_bbt_accepts_hf_padding_mask_without_changing_shared_scores():
    torch.manual_seed(19)
    module = GPT2BBTAttention(_config(), latent_rank=6).eval()
    hidden_states = torch.randn(2, 4, 8)
    keep, additive_mask = _padding_mask()

    with torch.no_grad():
        output, _, weights = module(
            hidden_states, attention_mask=additive_mask, output_attentions=True
        )

        latent = module.basis(hidden_states)
        scores = (latent @ latent.transpose(-2, -1)) / (6**0.5)
        causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        scores = scores + additive_mask[:, 0]
        probabilities = torch.softmax(scores, dim=-1)
        value = module.v_proj(hidden_states).view(2, 4, 2, 4).transpose(1, 2)
        expected = (probabilities.unsqueeze(1) @ value).transpose(1, 2).reshape(2, 4, 8)
        expected = module.o_proj(expected)

    assert output.shape == (2, 4, 8)
    assert weights.shape == (2, 2, 4, 4)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights[:, 0], weights[:, 1])
    allowed = causal[None, None] & keep[:, None, None, :]
    assert torch.count_nonzero(weights.masked_select(~allowed.expand_as(weights))) == 0


def test_bbt_accepts_boolean_mask_and_rejects_head_specific_scores():
    torch.manual_seed(23)
    module = GPT2BBTAttention(_config(), latent_rank=6).eval()
    hidden_states = torch.randn(2, 4, 8)
    keep, additive_mask = _padding_mask()
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    boolean_mask = causal[None, None] & keep[:, None, None, :]

    with torch.no_grad():
        additive_output, _, _ = module(hidden_states, attention_mask=additive_mask)
        boolean_output, _, weights = module(
            hidden_states, attention_mask=boolean_mask, output_attentions=True
        )

    torch.testing.assert_close(boolean_output, additive_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(weights[:, 0], weights[:, 1])
    with pytest.raises(ValueError, match="head dimension"):
        module(hidden_states, attention_mask=torch.zeros(2, 2, 1, 4))
