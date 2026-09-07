"""
Model factory: loads models.yaml, resolves env vars, instantiates adapters.

Cloud model adapters (openai, gemini, fireworks, claude) are imported lazily so their SDKs
don't need to be installed when you're only running hf_local models.
"""

import os
import re
from typing import List

import yaml

from .base import BaseModel, ModelConfig


def _get_class(model_type: str, model_name: str):
    """Return the adapter class for model_type, importing lazily."""
    if model_type == "hf_local":
        try:
            from .hf_local_model import HFLocalModel
            return HFLocalModel
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires torch and transformers. "
                "Install them with: pip install torch transformers accelerate sentencepiece"
            )

    if model_type == "openai":
        try:
            from .openai_model import OpenAIModel
            return OpenAIModel
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires the openai package. "
                "Install it with: pip install openai"
            )

    if model_type == "gemini":
        try:
            from .gemini_model import GeminiModel
            return GeminiModel
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires the google-genai package. "
                "Install it with: pip install google-genai"
            )

    if model_type == "fireworks":
        try:
            from .fireworks_model import FireworksModel
            return FireworksModel
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires the openai package. "
                "Install it with: pip install openai"
            )

    if model_type in {"claude", "anthropic"}:
        try:
            from .claude_model import ClaudeModel
            return ClaudeModel
        except ImportError:
            raise ImportError(
                f"Model '{model_name}' requires the anthropic package. "
                "Install it with: pip install anthropic"
            )

    raise ValueError(
        f"Unknown model type '{model_type}' for model '{model_name}'. "
        f"Valid types: hf_local, openai, gemini, fireworks, claude"
    )


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def load_models_from_config(config_path: str) -> List[BaseModel]:
    """
    Parse models.yaml and return a list of instantiated model adapters.
    Model-level settings override the global defaults.
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    models: List[BaseModel] = []

    for m in raw["models"]:
        merged = {**defaults, **m}

        api_key_raw = merged.get("api_key", "")
        api_key = _resolve_env_vars(str(api_key_raw)) if api_key_raw else None

        config = ModelConfig(
            name=merged["name"],
            type=merged["type"],
            model_id=merged["model_id"],
            model_path=merged.get("model_path"),
            api_key=api_key,
            base_url=merged.get("base_url"),
            max_tokens=int(merged.get("max_tokens", 512)),
            temperature=float(merged.get("temperature", 0.0)),
            timeout=int(merged.get("timeout", 120)),
        )

        cls = _get_class(config.type, config.name)
        models.append(cls(config))

    return models
