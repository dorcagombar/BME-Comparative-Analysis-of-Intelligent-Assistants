"""Abstract base class and shared data models for all LLM adapters."""

import time
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


class BaseModel(ABC):
    """Common interface for all model adapters — cloud and local."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name

    def load(self) -> None:
        """No-op for cloud models; HFLocalModel overrides to load weights."""

    def unload(self) -> None:
        """No-op for cloud models; HFLocalModel overrides to free GPU memory."""

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

    @abstractmethod
    def generate(self, prompt: str) -> ModelResponse:
        """Send a text prompt and return the response. Must never raise — use error field instead."""
        ...
