"""Llama-style GQA and GQA+BMB-UV models used in the FineWeb-Edu experiments.

The non-QK topology follows the public TPA nanoGPT recipe: tied GPT-2
embeddings, RoPE, parameter-free RMS normalization, SwiGLU, and no biases.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from bmb.attention import (
    BMBAttention,
    RotaryEmbedding,
    apply_rope,
    grouped_attention,
)


class BMBConfig(PretrainedConfig):
    model_type = "bmb-gqa"

    def __init__(
        self,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 22,
        n_kv_head: int = 2,
        n_embd: int = 768,
        head_dim: int = 64,
        value_dim: int = 64,
        intermediate_size: int | None = None,
        latent_rank: int = 64,
        factor_rank: int = 32,
        attention_type: str = "bmbuv",
        block_size: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        attention_backend: str = "sdpa",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        # Standard Transformers aliases used by generation and cache utilities.
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.hidden_size = n_embd
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.intermediate_size = intermediate_size or math.floor(8 * n_embd / 3)
        self.latent_rank = latent_rank
        self.factor_rank = factor_rank
        self.attention_type = attention_type
        self.block_size = block_size
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.attention_backend = attention_backend
        if n_head % n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if attention_type not in {"gqa", "bmbuv"}:
            raise ValueError("attention_type must be 'gqa' or 'bmbuv'")
        if attention_backend not in {"sdpa", "fa4"}:
            raise ValueError("attention_backend must be 'sdpa' or 'fa4'")


class GQABase(nn.Module):
    def __init__(self, config: BMBConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.group_size = config.n_head // config.n_kv_head
        self.value_dim = config.value_dim
        self.v_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.value_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.n_head * config.value_dim, config.n_embd, bias=False
        )
        self.rotary = RotaryEmbedding(self.score_dim, config.rope_theta)

    @property
    def score_dim(self) -> int:
        raise NotImplementedError

    def project_qk(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
    ):
        batch, query_len, _ = x.shape
        q, k = self.project_qk(x)
        v = self.v_proj(x).view(
            batch, query_len, self.n_kv_head, self.value_dim
        )
        if past_key_value is None:
            past_len = 0
        elif isinstance(past_key_value, dict):
            past_len = past_key_value["length"]
        else:
            past_len = past_key_value[0].shape[2]
        cos, sin = self.rotary(
            x, past_len, query_len, position_ids=position_ids
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if isinstance(past_key_value, dict):
            # Static-cache path used by the systems benchmark. The dictionary
            # owns preallocated K/V tensors and the current sequence length.
            past_key_value["k"][:, :, past_len:past_len + query_len] = k
            past_key_value["v"][:, :, past_len:past_len + query_len] = v
            k = past_key_value["k"][:, :, :past_len + query_len]
            v = past_key_value["v"][:, :, :past_len + query_len]
            past_key_value["length"] = past_len + query_len
            present = past_key_value if use_cache else None
        else:
            if past_key_value is not None:
                k = torch.cat((past_key_value[0], k), dim=2)
                v = torch.cat((past_key_value[1], v), dim=2)
            present = (k, v) if use_cache else None
        y = grouped_attention(
            q,
            k,
            v,
            is_causal=True,
            attention_mask=attention_mask,
            past_len=past_len,
            backend=self.config.attention_backend,
        )
        y = y.transpose(1, 2).contiguous().view(
            batch, query_len, self.n_head * self.value_dim
        )
        return self.o_proj(y), present


class GQAAttention(GQABase):
    """Dense grouped-query attention with independent Q/K projections."""

    def __init__(self, config: BMBConfig):
        self.head_dim = config.head_dim
        super().__init__(config)
        self.q_proj = nn.Linear(
            config.n_embd, config.n_head * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.n_embd, config.n_kv_head * config.head_dim, bias=False
        )

    @property
    def score_dim(self) -> int:
        return self.head_dim

    def project_qk(self, x: torch.Tensor):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.n_kv_head, self.head_dim)
        return q, k


class BMBUVAttention(BMBAttention):
    """Paper-model adapter for the standalone :class:`BMBAttention` module.

    Parameter names and the explicit position-encoded K/V cache layout are
    unchanged, so FineWeb checkpoints and the plug-and-play module share one
    implementation rather than maintaining two independent definitions.
    """

    def __init__(self, config: BMBConfig):
        super().__init__(
            d_model=config.n_embd,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            latent_rank=config.latent_rank,
            factor_rank=config.factor_rank,
            value_dim=config.value_dim,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
            backend=config.attention_backend,
        )
        self.config = config


class SwiGLU(nn.Module):
    def __init__(self, config: BMBConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.up = nn.Linear(config.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: BMBConfig):
        super().__init__()
        attention_cls = BMBUVAttention if config.attention_type == "bmbuv" else GQAAttention
        self.attn = attention_cls(config)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
    ):
        attn, present = self.attn(
            F.rms_norm(x, (x.shape[-1],)),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        x = x + attn
        x = x + self.mlp(F.rms_norm(x, (x.shape[-1],)))
        return x, present


class BMBForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = BMBConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BMBConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self.post_init()
        self._reset_experiment_initialization()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _reset_experiment_initialization(self):
        """Restore the paper's Xavier factors and scaled residual projections."""
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for layer in self.layers:
            attention = layer.attn
            if isinstance(attention, BMBUVAttention):
                nn.init.xavier_uniform_(attention.basis.weight)
                for factors in (attention.q_factor, attention.k_factor):
                    for head_factor in factors:
                        nn.init.xavier_uniform_(head_factor)
            nn.init.normal_(attention.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down.weight, mean=0.0, std=residual_std)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        self.lm_head.weight = value.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value
        self.embed_tokens.weight = value.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | tuple | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        dynamic_cache_input = isinstance(past_key_values, DynamicCache)
        static_cache_input = isinstance(past_key_values, list) and all(
            isinstance(layer, dict) for layer in past_key_values
        )
        if dynamic_cache_input:
            past_key_values = past_key_values.to_legacy_cache()

        x = self.embed_tokens(input_ids)
        presents = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            if isinstance(past, (tuple, list)) and past and past[0] is None:
                # A new Transformers DynamicCache starts with empty layers.
                past = None
            if self.training and self.gradient_checkpointing and not use_cache:
                x, present = self._gradient_checkpointing_func(
                    layer.__call__, x, attention_mask, past, use_cache, position_ids
                )
            else:
                x, present = layer(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=past,
                    use_cache=use_cache,
                    position_ids=position_ids,
                )
            if use_cache:
                presents.append(present)
        x = F.rms_norm(x, (x.shape[-1],))
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(x).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )
        if presents is not None and not static_cache_input:
            cache = DynamicCache()
            for index, (key, value) in enumerate(presents):
                cache.update(key, value, index)
            presents = cache
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=presents
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif len(past_key_values) and past_key_values[0] is not None:
                past_length = past_key_values[0][0].shape[2]
        if past_length:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
