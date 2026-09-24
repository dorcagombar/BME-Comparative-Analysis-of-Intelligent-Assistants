"""Adapter for Google Gemini via the google-genai SDK (v1+), synchronous."""

import time
from pathlib import Path

from google import genai
from google.genai import types

from .base import BaseModel, ModelConfig, ModelResponse

_AUDIO_MIME_MAP = {
    ".wav":  "audio/wav",
    ".wave": "audio/wav",
    ".mp3":  "audio/mp3",
    ".m4a":  "audio/mp4",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class GeminiModel(BaseModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = genai.Client(api_key=config.api_key)
        self._model_id = config.model_id

    def _generation_config(self, structured: bool = False):
        """Build settings per call so runtime UI changes are respected."""
        max_tokens = self.resolve_max_tokens(structured=structured)
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        thinking_level = str(extra.get("thinking_level") or "").strip().upper()
        kwargs = {
            "max_output_tokens": max_tokens,
            "temperature": float(self.config.temperature),
        }
        if thinking_level:
            valid_levels = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
            if thinking_level not in valid_levels:
                raise ValueError(
                    "Gemini extra.thinking_level must be one of: "
                    + ", ".join(sorted(valid_levels))
                )
            if not hasattr(types, "ThinkingConfig") or not hasattr(types, "ThinkingLevel"):
                raise RuntimeError(
                    "The installed google-genai package does not support "
                    "thinking_level. Upgrade google-genai before running Gemini 3."
                )
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=getattr(types.ThinkingLevel, thinking_level)
            )
        if structured:
            kwargs["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kwargs)

    def _thinking_level(self):
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        value = str(extra.get("thinking_level") or "").strip().lower()
        return value or None

    def supports_audio(self) -> bool:
        return True

    def generate(self, prompt: str) -> ModelResponse:
        return self._generate(prompt, structured=False)

    def generate_structured(self, prompt: str) -> ModelResponse:
        return self._generate(prompt, structured=True)

    @staticmethod
    def _structured_output_unsupported(error: Exception) -> bool:
        text = str(error).lower()
        return (
            "response_mime_type" in text
            or "response mime" in text
            or ("json" in text and ("unsupported" in text or "not support" in text))
        )

    def _generate(self, prompt: str, structured: bool) -> ModelResponse:
        start = time.perf_counter()
        try:
            native_structured = structured
            try:
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config=self._generation_config(structured=structured),
                )
            except Exception as exc:
                if not structured or not self._structured_output_unsupported(exc):
                    raise
                native_structured = False
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=prompt,
                    config=self._generation_config(structured=False),
                )
            # Extract text and finish_reason from first candidate
            text, finish_reason, ratings_str = "", "", ""
            try:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name
                ratings_str = ", ".join(
                    f"{r.category.name}={r.probability.name}"
                    for r in (candidate.safety_ratings or [])
                )
                # Collect partial text from parts if response.text is None
                text = response.text or "".join(
                    p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
                )
            except Exception:
                text = response.text or ""

            error = None
            if finish_reason == "MAX_TOKENS":
                error = (
                    "Truncated at max_output_tokens="
                    f"{self.resolve_max_tokens(structured=structured)} "
                    "(finish_reason=MAX_TOKENS, "
                    f"thinking_level={self._thinking_level() or 'provider default'})."
                )
            elif not text:
                detail = finish_reason or "unknown"
                if ratings_str:
                    detail += f" [{ratings_str}]"
                error = f"No text returned (finish_reason={detail})"

            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=text.strip() if text else "",
                latency_seconds=time.perf_counter() - start,
                error=error,
                finish_reason=finish_reason or None,
                prompt_tokens=getattr(getattr(response, "usage_metadata", None), "prompt_token_count", None),
                completion_tokens=getattr(getattr(response, "usage_metadata", None), "candidates_token_count", None),
                metadata={
                    "requested_max_tokens": self.resolve_max_tokens(structured=structured),
                    "thinking_level": self._thinking_level(),
                    "thought_tokens": getattr(
                        getattr(response, "usage_metadata", None),
                        "thoughts_token_count",
                        None,
                    ),
                    "structured": structured,
                    "native_structured_output": native_structured,
                },
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=str(e),
                metadata={
                    "structured": structured,
                    "thinking_level": self._thinking_level(),
                    "requested_max_tokens": self.resolve_max_tokens(
                        structured=structured
                    ),
                },
            )

    def generate_audio(self, audio_path: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            suffix = Path(audio_path).suffix.lower()
            mime = _AUDIO_MIME_MAP.get(suffix, "audio/wav")
            audio_bytes = Path(audio_path).read_bytes()

            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
                config=self._generation_config(structured=False),
            )
            text = response.text
            if text is None:
                reason = ""
                try:
                    reason = response.candidates[0].finish_reason.name
                except Exception:
                    pass
                return ModelResponse(
                    model_name=self.name,
                    question=Path(audio_path).name,
                    prediction="",
                    latency_seconds=time.perf_counter() - start,
                    error=f"No text returned (finish_reason={reason or 'unknown'})",
                    finish_reason=reason or None,
                )
            return ModelResponse(
                model_name=self.name,
                question=Path(audio_path).name,
                prediction=text.strip(),
                latency_seconds=time.perf_counter() - start,
                finish_reason=(
                    response.candidates[0].finish_reason.name
                    if getattr(response, "candidates", None) else None
                ),
                prompt_tokens=getattr(
                    getattr(response, "usage_metadata", None),
                    "prompt_token_count",
                    None,
                ),
                completion_tokens=getattr(
                    getattr(response, "usage_metadata", None),
                    "candidates_token_count",
                    None,
                ),
                metadata={
                    "requested_max_tokens": self.resolve_max_tokens(),
                    "thinking_level": self._thinking_level(),
                    "thought_tokens": getattr(
                        getattr(response, "usage_metadata", None),
                        "thoughts_token_count",
                        None,
                    ),
                },
            )
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                question=Path(audio_path).name,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=str(e),
            )
