"""Low-temperature JSON-oriented LLM adapter.

The agent can run without an API key by falling back to deterministic rules.
When Groq is configured, this client keeps reasoning constrained and asks for
machine-readable outputs only.
"""

from __future__ import annotations

import json
import os
from typing import Any

from app.core.config import settings


class LLMUnavailable(RuntimeError):
    """Raised when no configured LLM client is available."""


class GroqReasoningClient:
    def __init__(
        self,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client: Any | None = None
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq

                self._client = Groq(api_key=api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def json_call(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._client:
            return fallback

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=True, indent=2),
            },
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else fallback
        except Exception:
            return fallback
