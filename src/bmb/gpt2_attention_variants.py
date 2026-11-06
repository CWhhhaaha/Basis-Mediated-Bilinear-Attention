"""GPT-2 compatible attention variants for causal language modeling.

Each variant preserves the core projection logic from the ViT implementations
but adds causal masking and conforms to the GPT2Attention interface.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _causal_mask(
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a bottom-right aligned causal mask with True for visible keys."""
    return torch.tril(
        torch.ones((query_len, key_len), dtype=torch.bool, device=device),
        diagonal=key_len - query_len,
    )


def _merge_causal_attention_mask(
    attention_mask: torch.Tensor,
    query_len: int,
    key_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Combine causal and explicit masks for SDPA and the reference path."""
    causal = _causal_mask(query_len, key_len, device)
    mask = attention_mask.to(device=device)
    if mask.dtype == torch.bool:
        return mask & causal
    additive_causal = torch.zeros(
        (query_len, key_len), dtype=dtype, device=device
    ).masked_fill(~causal, float("-inf"))
    return mask.to(dtype=dtype) + additive_causal


def _shared_score_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Remove a singleton head axis for BBT's head-shared attention scores."""
    if attention_mask.ndim == 4:
        if attention_mask.shape[1] != 1:
            raise ValueError(
                "BBT uses one attention-score matrix per batch item; "
                "the head dimension of attention_mask must be 1"
            )
        return attention_mask[:, 0]
    if attention_mask.ndim == 3:
        return attention_mask
    if attention_mask.ndim == 2:
        return attention_mask[:, None, :]
    raise ValueError("attention_mask must have shape [B,L], [B,T,L], or [B,1,T,L]")


def _layer_past(past_key_values: Any | None, kwargs: dict[str, Any]):
    """Accept both the custom argument name and GPT2Block's layer_past."""
    if past_key_values is None:
        return kwargs.get("layer_past")
    return past_key_values


def _is_transformers_cache(past_key_values: Any | None) -> bool:
    return past_key_values is not None and hasattr(past_key_values, "update")


def _append_key_value_cache(
    k, v, past_key_values, use_cache: bool, layer_idx: int | None
):
    if _is_transformers_cache(past_key_values):
        k, v = past_key_values.update(k, v, layer_idx)
        present = past_key_values if use_cache else None
    else:
        if past_key_values is not None:
            k = torch.cat((past_key_values[0], k), dim=2)
            v = torch.cat((past_key_values[1], v), dim=2)
        present = (k, v) if use_cache else None
    return k, v, present


def _append_latent_value_cache(
    latent, v, past_key_values, use_cache: bool, layer_idx: int | None
):
    if _is_transformers_cache(past_key_values):
        # DynamicCache concatenates on the sequence axis (-2). Native BMB and
        # BBT cache [B,T,r] latents, so expose a singleton head axis while
        # updating the cache and remove it for the score computation.
        key_latent, v = past_key_values.update(latent.unsqueeze(1), v, layer_idx)
        key_latent = key_latent.squeeze(1)
        present = past_key_values if use_cache else None
    else:
        key_latent = (
            latent
            if past_key_values is None
            else torch.cat((past_key_values[0], latent), dim=1)
        )
        if past_key_values is not None:
            v = torch.cat((past_key_values[1], v), dim=2)
        present = (key_latent, v) if use_cache else None
    return key_latent, v, present


def _attention_output(y, present, att, kwargs: dict[str, Any]):
    # Transformers 4.56 GPT2Block returns (attention, weights) from its
    # attention module and owns the DynamicCache. Earlier blocks expect
    # (attention, present, weights). cache_position identifies the new call.
    if "cache_position" in kwargs:
        return y, att
    return y, present, att


def _sdpa_or_manual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale_dim: int,
    attn_dropout: nn.Dropout,
    training: bool,
    attention_mask: torch.Tensor | None,
    output_attentions: bool,
):
    """Use PyTorch SDPA when possible, otherwise fall back to explicit attention."""
    mask = None
    if attention_mask is not None:
        mask = _merge_causal_attention_mask(
            attention_mask, q.size(-2), k.size(-2), q.device, q.dtype
        )
    elif q.size(-2) != k.size(-2):
        # SDPA's built-in causal mask is top-left aligned for non-square inputs.
        mask = _causal_mask(q.size(-2), k.size(-2), q.device)

    if not output_attentions:
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=attn_dropout.p if training else 0.0,
            is_causal=mask is None,
        )
        return y, None

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(scale_dim))
    if mask is None:
        causal = _causal_mask(q.size(-2), k.size(-2), q.device)
        att = att.masked_fill(~causal, float("-inf"))
    elif mask.dtype == torch.bool:
        att = att.masked_fill(~mask, float("-inf"))
    else:
        att = att + mask
    att = torch.softmax(att, dim=-1)
    att = attn_dropout(att)
    y = att @ v
    return y, att


class GPT2LowRankAttention(nn.Module):
    """Low-rank Q/K for GPT-2 causal LM.

    Q = X @ (A^Q @ B^Q), K = X @ (A^K @ B^K) per head.
    V and output projection remain standard.
    """

    def __init__(self, config, low_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.low_rank = int(low_rank)
        self.layer_idx = layer_idx

        # Head-wise low-rank factors: A in R^{H, d, r}, B in R^{H, r, d_k}
        self.q_a = nn.Parameter(torch.empty(self.n_head, self.n_embd, self.low_rank))
        self.q_b = nn.Parameter(torch.empty(self.n_head, self.low_rank, self.head_dim))
        self.k_a = nn.Parameter(torch.empty(self.n_head, self.n_embd, self.low_rank))
        self.k_b = nn.Parameter(torch.empty(self.n_head, self.low_rank, self.head_dim))

        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask buffer
        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        for p in (self.q_a, self.q_b, self.k_a, self.k_b):
            nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def _project(self, x: torch.Tensor, a: nn.Parameter, b: nn.Parameter) -> torch.Tensor:
        """X @ (A @ B) via einsum: [B,T,d] x [H,d,r] x [H,r,d_k] -> [B,H,T,d_k]"""
        mid = torch.einsum("btd,hdr->bhtr", x, a)
        out = torch.einsum("bhtr,hrd->bhtd", mid, b)
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        q = self._project(hidden_states, self.q_a, self.q_b)
        k = self._project(hidden_states, self.k_a, self.k_b)
        v = self.v_proj(hidden_states)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v, present = _append_key_value_cache(
            k, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        y, att = _sdpa_or_manual(
            q,
            k,
            v,
            scale_dim=self.head_dim,
            attn_dropout=self.attn_dropout,
            training=self.training,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return _attention_output(y, present, att, kwargs)


class GPT2FullySharedAttention(nn.Module):
    """Reformer-style fully shared Q=K for GPT-2 causal LM."""

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.layer_idx = layer_idx

        self.query_key = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        nn.init.normal_(self.query_key.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        for m in (self.query_key, self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        q = self.query_key(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.query_key(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v, present = _append_key_value_cache(
            k, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        y, att = _sdpa_or_manual(
            q,
            k,
            v,
            scale_dim=self.head_dim,
            attn_dropout=self.attn_dropout,
            training=self.training,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return _attention_output(y, present, att, kwargs)


class GPT2UVLatentAttention(nn.Module):
    """BMB-UV latent attention for GPT-2 causal LM.

    Shared latent basis Z = X @ B, then head-specific UV factors:
        Q = einsum(Z, U_factor), K = einsum(Z, V_factor)
    """

    def __init__(
        self,
        config,
        latent_rank: int,
        latent_factor_rank: int | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.latent_factor_rank = int(latent_factor_rank if latent_factor_rank is not None else latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.u_factor = nn.Parameter(torch.empty(self.n_head, self.latent_rank, self.latent_factor_rank))
        self.v_factor = nn.Parameter(torch.empty(self.n_head, self.latent_rank, self.latent_factor_rank))
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        nn.init.xavier_uniform_(self.basis.weight)
        for factor in (self.u_factor, self.v_factor):
            for head_factor in factor:
                nn.init.xavier_uniform_(head_factor)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        for m in (self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)  # [B, T, r]
        q = torch.einsum("btr,hrs->bhts", latent, self.u_factor)
        k = torch.einsum("btr,hrs->bhts", latent, self.v_factor)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k, v, present = _append_key_value_cache(
            k, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        y, att = _sdpa_or_manual(
            q,
            k,
            v,
            scale_dim=self.latent_factor_rank,
            attn_dropout=self.attn_dropout,
            training=self.training,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return _attention_output(y, present, att, kwargs)


class GPT2BBTAttention(nn.Module):
    """BBT attention for GPT-2 causal LM.

    A shared latent basis forms identical query/key coordinates for all heads.
    Head diversity is carried only by the value pathway.
    """

    def __init__(self, config, latent_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        nn.init.normal_(self.basis.weight, mean=0.0, std=std)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        for m in (self.v_proj, self.o_proj):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        key_latent, v, present = _append_latent_value_cache(
            latent, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        att = (latent @ key_latent.transpose(-2, -1)) * (1.0 / math.sqrt(self.latent_rank))
        causal = _causal_mask(T, key_latent.size(1), att.device)
        att = att.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            mask = _shared_score_attention_mask(attention_mask).to(device=att.device)
            if mask.dtype == torch.bool:
                att = att.masked_fill(~mask, float("-inf"))
            else:
                att = att + mask.to(dtype=att.dtype)
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att.unsqueeze(1) @ v
        att = att.unsqueeze(1).expand(-1, self.n_head, -1, -1) if output_attentions else None
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return _attention_output(y, present, att, kwargs)


class GPT2SymmetricLatentAttention(nn.Module):
    """BMB attention for GPT-2 causal LM.

    Heads share a latent basis but use head-specific bilinear interaction
    operators inside that basis.
    """

    def __init__(self, config, latent_rank: int, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.latent_rank = int(latent_rank)
        self.layer_idx = layer_idx

        self.basis = nn.Linear(self.n_embd, self.latent_rank, bias=False)
        self.core = nn.Parameter(torch.empty(self.latent_rank, self.latent_rank))
        self.head_residual = nn.Parameter(
            torch.empty(self.n_head, self.latent_rank, self.latent_rank)
        )
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        nn.init.xavier_uniform_(self.basis.weight)
        nn.init.xavier_uniform_(self.core)
        for head_residual in self.head_residual:
            nn.init.xavier_uniform_(head_residual)
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def head_matrices(self) -> torch.Tensor:
        centered = self.head_residual - self.head_residual.mean(dim=0, keepdim=True)
        return self.core.unsqueeze(0) / float(self.n_head) + centered

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        B, T, C = hidden_states.size()

        latent = self.basis(hidden_states)  # [B, T, r]
        v = self.v_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        key_latent, v, present = _append_latent_value_cache(
            latent, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        head_mats = self.head_matrices()  # [H, r, r]
        att = torch.einsum("btr,hrs,bus->bhtu", latent, head_mats, key_latent)
        att = att * (1.0 / math.sqrt(self.latent_rank))
        causal = _causal_mask(T, key_latent.size(1), att.device)
        att = att.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            mask = attention_mask.to(device=att.device)
            if mask.dtype == torch.bool:
                att = att.masked_fill(~mask, float("-inf"))
            else:
                att = att + mask.to(dtype=att.dtype)

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))

        return _attention_output(
            y, present, att if output_attentions else None, kwargs
        )


class GPT2PartialSharedAttention(nn.Module):
    """Layer-level partial Q/K sharing followed by the usual head split."""

    def __init__(self, config, shared_qk_dim: int, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.shared_qk_dim = int(shared_qk_dim)
        self.private_qk_dim = self.n_embd - self.shared_qk_dim
        if not 0 <= self.private_qk_dim <= self.n_embd:
            raise ValueError(f"shared_qk_dim must be between 0 and {self.n_embd}")
        self.layer_idx = layer_idx

        self.share = nn.Linear(self.n_embd, self.shared_qk_dim, bias=True)
        if self.private_qk_dim > 0:
            self.query_priv = nn.Linear(self.n_embd, self.private_qk_dim, bias=True)
            self.key_priv = nn.Linear(self.n_embd, self.private_qk_dim, bias=True)
        else:
            self.query_priv = None
            self.key_priv = None
        self.v_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self._reset_parameters()

    def _reset_parameters(self):
        std = float(self.config.initializer_range)
        modules = [self.share, self.v_proj]
        if self.query_priv is not None:
            modules.extend([self.query_priv, self.key_priv])
        for module in modules:
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=std / math.sqrt(2.0 * self.config.n_layer))
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        b, t, c = hidden_states.size()
        shared = self.share(hidden_states)
        if self.private_qk_dim > 0:
            query = torch.cat([shared, self.query_priv(hidden_states)], dim=-1)
            key = torch.cat([shared, self.key_priv(hidden_states)], dim=-1)
        else:
            query = shared
            key = shared
        q = self._split_heads(query)
        k = self._split_heads(key)
        v = self._split_heads(self.v_proj(hidden_states))
        k, v, present = _append_key_value_cache(
            k, v, _layer_past(past_key_values, kwargs), use_cache, self.layer_idx
        )

        y, att = _sdpa_or_manual(
            q,
            k,
            v,
            scale_dim=self.head_dim,
            attn_dropout=self.attn_dropout,
            training=self.training,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return _attention_output(
            self.resid_dropout(self.o_proj(y)), present, att, kwargs
        )


def replace_gpt2_attention(model: nn.Module, variant: str, **kwargs: Any) -> None:
    """In-place replace attention layers in a GPT-2 model."""
    for i, block in enumerate(model.transformer.h):
        if variant == "lowrank":
            block.attn = GPT2LowRankAttention(model.config, low_rank=kwargs["rank"], layer_idx=i)
        elif variant == "fullyshared":
            block.attn = GPT2FullySharedAttention(model.config, layer_idx=i)
        elif variant == "bbt":
            block.attn = GPT2BBTAttention(model.config, latent_rank=kwargs["rank"], layer_idx=i)
        elif variant == "bmb":
            block.attn = GPT2SymmetricLatentAttention(
                model.config, latent_rank=kwargs["rank"], layer_idx=i
            )
        elif variant == "bmbuv":
            block.attn = GPT2UVLatentAttention(
                model.config,
                latent_rank=kwargs["rank"],
                latent_factor_rank=kwargs.get("factor_rank"),
                layer_idx=i,
            )
        elif variant == "partialshared":
            block.attn = GPT2PartialSharedAttention(
                model.config, shared_qk_dim=kwargs["shared_dim"], layer_idx=i
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")
