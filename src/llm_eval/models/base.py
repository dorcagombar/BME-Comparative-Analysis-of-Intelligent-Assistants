"""Abstract base class and shared data models for all LLM adapters."""

import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model instance."""
    name: str
    type: str           # "hf_local" | "openai" | "gemini" | "fireworks" | "claude"
    model_id: str
    # For hf_local: path to the downloaded snapshot directory on disk.
    # For cloud APIs: unused (set to same as model_id for consistency).
    model_path: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: int = 120
    extra: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Output from a single model.generate() call."""
    model_name: str
    question: str
    prediction: str
    latency_seconds: float
    error: Optional[str] = None   # non-None means the call failed
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: dict = field(default_factory=dict)


class BaseModel(ABC):
    """Common interface for all model adapters — cloud and local."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name

    def load(self) -> None:
        """No-op for cloud models; HFLocalModel overrides to load weights."""

    def unload(self) -> None:
        """No-op for cloud models; HFLocalModel overrides to free GPU memory."""

    def output_token_limit(self, *, structured: bool = False) -> int:
        """Return a conservative, configurable output-token ceiling.

        The benchmark uses one declared output ceiling for every model so
        token availability does not vary silently across systems. A model
        entry may override it only when a provider has a documented lower
        limit, using
        ``extra.max_output_tokens_limit`` or
        ``extra.structured_max_output_tokens_limit``.
        """
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        key = (
            "structured_max_output_tokens_limit"
            if structured else "max_output_tokens_limit"
        )
        benchmark_default = 16384
        raw = extra.get(
            key,
            extra.get("max_output_tokens_limit", benchmark_default),
        )
        try:
            limit = int(raw)
        except (TypeError, ValueError):
            limit = benchmark_default
        return max(1, limit)

    def resolve_max_tokens(
        self,
        requested: Optional[int] = None,
        *,
        structured: bool = False,
    ) -> int:
        """Validate and clamp an output-token request for this model."""
        raw = self.config.max_tokens if requested is None else requested
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_tokens must be a positive integer") from exc
        if value < 1:
            raise ValueError("max_tokens must be a positive integer")
        return min(value, self.output_token_limit(structured=structured))

    def configure_generation(self, max_tokens: int, temperature: float) -> int:
        """Apply validated common generation settings and return the token cap."""
        try:
            temp = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("temperature must be a finite number") from exc
        if not math.isfinite(temp) or temp < 0.0:
            raise ValueError("temperature must be a finite non-negative number")
        resolved = self.resolve_max_tokens(max_tokens)
        self.config.max_tokens = resolved
        self.config.temperature = temp
        return resolved

    def supports_audio(self) -> bool:
        """Return True if this model can process audio input."""
        return False

    def generate_audio(self, audio_path: str) -> "ModelResponse":
        """Default audio handler — returns an error. Override in audio-capable adapters."""
        return ModelResponse(
            model_name=self.name,
            question=audio_path,
            prediction="",
            latency_seconds=0.0,
            error=f"{self.name} does not support audio input",
        )

    def generate_structured(self, prompt: str) -> "ModelResponse":
        """Generate machine-readable output when the provider supports it.

        Adapters with native JSON-output controls override this method. The
        default preserves compatibility for providers that only expose normal
        text generation; the runner still validates the returned schema.
        """
        return self.generate(prompt)

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        """Send a text prompt and return the response. Must never raise — use error field instead."""
        ...
