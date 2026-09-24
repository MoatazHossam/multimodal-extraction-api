import json
from typing import Any

import httpx

from app.ai.provider import AIProvider, AIProviderError, AIProviderTimeoutError


class OllamaProvider(AIProvider):
    """Structured extraction implementation using Ollama's local HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_seconds)
        )

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "stream": False,
            "think": False,
            "format": output_schema,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "options": {"temperature": 0},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise AIProviderTimeoutError("Ollama request timed out") from exc
        except httpx.HTTPError as exc:
            raise AIProviderError("Ollama request failed") from exc

        try:
            body = response.json()
            content = body["message"]["content"]
            result = json.loads(content)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AIProviderError("Ollama returned an invalid structured response") from exc

        if not isinstance(result, dict):
            raise AIProviderError("Ollama structured response must be a JSON object")
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

