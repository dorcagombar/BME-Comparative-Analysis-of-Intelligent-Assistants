"""Adapter for Fireworks AI (OpenAI-compatible API), synchronous."""

import time

from openai import OpenAI

from .base import BaseModel, ModelConfig, ModelResponse

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


class FireworksModel(BaseModel):
    """Calls Fireworks AI via the OpenAI-compatible endpoint."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or FIREWORKS_BASE_URL,
            timeout=config.timeout,
        )

    def generate(self, prompt: str) -> ModelResponse:
        return self._generate(prompt, structured=False)

    def generate_structured(self, prompt: str) -> ModelResponse:
        return self._generate(prompt, structured=True)

    @staticmethod
    def _message_text(message) -> str:
        """Normalize OpenAI-compatible string or multipart message content."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text")
                else:
                    value = getattr(item, "text", None)
                if value:
                    parts.append(str(value))
            return "".join(parts).strip()
        return ""

    @staticmethod
    def _structured_output_unsupported(error: Exception) -> bool:
        text = str(error).lower()
        return (
            "response_format" in text
            or "json_object" in text
            or ("json" in text and ("unsupported" in text or "not support" in text))
        )

    def _generate(self, prompt: str, structured: bool) -> ModelResponse:
        start = time.perf_counter()
        try:
            max_tokens = self.resolve_max_tokens(structured=structured)
            request = dict(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.config.temperature,
            )
            if structured:
                request["response_format"] = {"type": "json_object"}
            native_structured = structured
            try:
                response = self._client.chat.completions.create(**request)
            except Exception as exc:
                if not structured or not self._structured_output_unsupported(exc):
                    raise
                # Some OpenAI-compatible models do not implement response_format.
                # Retry only that capability mismatch; the prompt and downstream
                # schema validation still enforce machine-readable output.
                request.pop("response_format", None)
                native_structured = False
                response = self._client.chat.completions.create(**request)
            if not response.choices:
                raise RuntimeError("Provider returned no completion choices.")
            choice = response.choices[0]
            prediction = self._message_text(choice.message)
            finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
            usage = getattr(response, "usage", None)
            error = None
            if finish_reason in {"length", "max_tokens"}:
                error = (
                    f"Truncated at max_tokens (finish_reason={finish_reason})"
                )
            elif not prediction:
                has_reasoning = bool(
                    getattr(choice.message, "reasoning_content", None)
                    or getattr(choice.message, "reasoning", None)
                )
                detail = "; reasoning was returned without a final answer" if has_reasoning else ""
                error = f"No final text returned (finish_reason={finish_reason}{detail})"
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=prediction,
                latency_seconds=time.perf_counter() - start,
                error=error,
                finish_reason=finish_reason,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                metadata={
                    "requested_max_tokens": max_tokens,
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
                metadata={"structured": structured},
            )
