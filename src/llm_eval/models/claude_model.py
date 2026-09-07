"""Adapter for Claude via the Anthropic API (synchronous)."""

import time

from anthropic import Anthropic

from .base import BaseModel, ModelConfig, ModelResponse


class ClaudeModel(BaseModel):
    """Calls the Anthropic Messages API synchronously."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        client_options = {
            "api_key": config.api_key,
            "timeout": config.timeout,
        }

        if config.base_url:
            client_options["base_url"] = config.base_url

        self._client = Anthropic(**client_options)

    def generate(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()

        try:
            response = self._client.messages.create(
                model=self.config.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
            )

            prediction = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip()

            error = None

            if response.stop_reason == "max_tokens":
                error = (
                    "Truncated at max_tokens "
                    "(stop_reason=max_tokens) — "
                    "increase max_tokens in config"
                )
            elif not prediction:
                error = (
                    f"No text returned "
                    f"(stop_reason={response.stop_reason or 'unknown'})"
                )

            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction=prediction,
                latency_seconds=time.perf_counter() - start,
                error=error,
            )

        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                question=prompt,
                prediction="",
                latency_seconds=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )
