"""Synchronous Anthropic Claude adapter with reproducible effort controls."""

import time

from anthropic import Anthropic

from .base import BaseModel, ModelConfig, ModelResponse


class ClaudeModel(BaseModel):
    """Call Claude through the Anthropic Messages API."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = Anthropic(
            api_key=config.api_key,
            timeout=config.timeout,
        )

    def generate(self, prompt: str) -> ModelResponse:
        return self._generate(prompt, structured=False)

    def generate_structured(self, prompt: str) -> ModelResponse:
        # The runner validates the complete JSON schema. Native structured
        # output is not assumed, keeping this adapter compatible with the
        # standard Anthropic Messages endpoint.
        return self._generate(prompt, structured=True)

    def _effort(self) -> str:
        extra = self.config.extra if isinstance(self.config.extra, dict) else {}
        effort = str(extra.get("effort") or "medium").strip().lower()
        valid = {"low", "medium", "high", "xhigh", "max"}
        if effort not in valid:
            raise ValueError(
                "Claude extra.effort must be one of: " + ", ".join(sorted(valid))
            )
        return effort

    @staticmethod
    def _text_content(response) -> str:
        parts = []
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if block_type == "text" and text:
                parts.append(str(text))
        return "".join(parts).strip()

    def _generate(self, prompt: str, structured: bool) -> ModelResponse:
        start = time.perf_counter()
        max_tokens = self.resolve_max_tokens(structured=structured)
        effort = self._effort()
        try:
            request = {
                "model": self.config.model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
                "output_config": {"effort": effort},
            }
            # Claude Sonnet 5 rejects non-default sampling parameters. Omit
            # temperature for that model; effort is the supported control.
            if not str(self.config.model_id).lower().startswith("claude-sonnet-5"):
                request["temperature"] = float(self.config.temperature)

            response = self._client.messages.create(**request)
            prediction = self._text_content(response)
            stop_reason = str(getattr(response, "stop_reason", "") or "unknown")
            usage = getattr(response, "usage", None)
            error = None
            if stop_reason == "max_tokens":
                error = (
                    f"Truncated at max_tokens={max_tokens} "
                    f"(stop_reason=max_tokens, effort={effort})."
                )
            elif not prediction:
                error = f"No text returned (stop_reason={stop_reason})."

            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=prediction,
                latency_seconds=time.perf_counter() - start,
                error=error,
                finish_reason=stop_reason,
                prompt_tokens=getattr(usage, "input_tokens", None),
                completion_tokens=getattr(usage, "output_tokens", None),
                metadata={
                    "requested_max_tokens": max_tokens,
                    "effort": effort,
                    "structured": structured,
                    "native_structured_output": False,
                },
            )
        except Exception as exc:
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
                metadata={
                    "requested_max_tokens": max_tokens,
                    "effort": effort,
                    "structured": structured,
                },
            )
