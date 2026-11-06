"""Register the included BMB model with Hugging Face Auto classes."""

from __future__ import annotations

_REGISTERED = False


def register_bmb():
    """Register once and return ``(BMBConfig, BMBForCausalLM)``."""
    global _REGISTERED
    from transformers import AutoConfig, AutoModelForCausalLM

    from bmb.llama_bmb import BMBConfig, BMBForCausalLM

    if not _REGISTERED:
        AutoConfig.register(BMBConfig.model_type, BMBConfig)
        AutoModelForCausalLM.register(BMBConfig, BMBForCausalLM)
        _REGISTERED = True
    return BMBConfig, BMBForCausalLM
