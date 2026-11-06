import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel

from bmb.gpt2_attention_variants import replace_gpt2_attention
from bmb.llama_bmb import BMBConfig, BMBForCausalLM


GPT2_VARIANTS = {
    "lowrank": {"rank": 4},
    "fullyshared": {},
    "bbt": {"rank": 6},
    "bmb": {"rank": 6},
    "bmbuv": {"rank": 6, "factor_rank": 3},
    "partialshared": {"shared_dim": 6},
}


def _gpt2_model(variant):
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=2,
        n_head=2,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    replace_gpt2_attention(model, variant, **GPT2_VARIANTS[variant])
    return model


@pytest.mark.parametrize("variant", GPT2_VARIANTS)
def test_gpt2_full_model_forward_backward_with_padding_mask(variant):
    torch.manual_seed(31)
    model = _gpt2_model(variant).train()
    input_ids = torch.randint(0, 32, (2, 5))
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = input_ids.masked_fill(attention_mask == 0, -100)

    output = model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    output.loss.backward()

    assert output.logits.shape == (2, 5, 32)
    assert torch.isfinite(output.loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.eval()
    with torch.no_grad():
        manual = model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    assert len(manual.attentions) == 2
    for weights in manual.attentions:
        assert weights.shape == (2, 2, 5, 5)
        assert torch.isfinite(weights).all()
        future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        assert torch.count_nonzero(weights.masked_select(future[None, None])) == 0
        if variant == "bbt":
            torch.testing.assert_close(weights[:, 0], weights[:, 1])


def _fineweb_model(attention_type):
    return BMBForCausalLM(
        BMBConfig(
            vocab_size=32,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=16,
            head_dim=4,
            value_dim=4,
            intermediate_size=32,
            latent_rank=8,
            factor_rank=4,
            attention_type=attention_type,
            block_size=16,
            dropout=0.0,
        )
    )


@pytest.mark.parametrize("attention_type", ["gqa", "bmbuv"])
def test_fineweb_default_and_checkpointed_training(attention_type):
    torch.manual_seed(37)
    model = _fineweb_model(attention_type).train()
    checkpointed = copy.deepcopy(model).train()
    checkpointed.gradient_checkpointing_enable()
    input_ids = torch.randint(0, 32, (2, 6))

    assert not model.gradient_checkpointing
    assert checkpointed.gradient_checkpointing
    reference = model(input_ids, labels=input_ids, use_cache=False)
    actual = checkpointed(input_ids, labels=input_ids, use_cache=False)
    reference.loss.backward()
    actual.loss.backward()

    torch.testing.assert_close(actual.logits, reference.logits, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.loss, reference.loss, atol=1e-6, rtol=1e-6)
    for (name, expected), (_, parameter) in zip(
        model.named_parameters(), checkpointed.named_parameters()
    ):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_fineweb_checkpointing_does_not_disable_cache():
    model = _fineweb_model("bmbuv")
    model.gradient_checkpointing_enable()
    model.train()
    input_ids = torch.randint(0, 32, (2, 6))

    output = model(input_ids, use_cache=True)

    assert output.past_key_values is not None
    assert len(output.past_key_values) == 2
    for key, value in output.past_key_values:
        assert key.shape == (2, 2, 6, 4)
        assert value.shape == (2, 2, 6, 4)


@pytest.mark.parametrize("variant", GPT2_VARIANTS)
def test_gpt2_incremental_cache_matches_full_forward(variant):
    torch.manual_seed(41)
    model = _gpt2_model(variant).eval()
    input_ids = torch.randint(0, 32, (2, 6))

    with torch.no_grad():
        full = model(input_ids, use_cache=False)
        prompt = model(input_ids[:, :-1], use_cache=True)
        decoded = model(
            input_ids[:, -1:],
            past_key_values=copy.deepcopy(prompt.past_key_values),
            use_cache=True,
        )
        decoded_manual = model(
            input_ids[:, -1:],
            past_key_values=prompt.past_key_values,
            output_attentions=True,
            use_cache=False,
        )

    assert prompt.past_key_values is not None
    assert len(prompt.past_key_values) == 2
    assert all(present is not None for present in prompt.past_key_values)
    torch.testing.assert_close(decoded.logits, full.logits[:, -1:], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(decoded_manual.logits, full.logits[:, -1:], atol=2e-6, rtol=2e-5)
    assert len(decoded_manual.attentions) == 2
    for weights in decoded_manual.attentions:
        assert weights.shape == (2, 2, 1, 6)
        if variant == "bbt":
            torch.testing.assert_close(weights[:, 0], weights[:, 1])


@pytest.mark.parametrize("attention_type", ["gqa", "bmbuv"])
def test_fineweb_generate_with_cache(attention_type):
    model = _fineweb_model(attention_type).eval()
    input_ids = torch.randint(0, 32, (2, 5))

    assert model.can_generate()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )

    assert generated.shape == (2, 8)
    torch.testing.assert_close(generated[:, :5], input_ids)
