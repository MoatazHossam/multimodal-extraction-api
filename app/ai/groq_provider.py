import json
from typing import Any

import httpx

from app.ai.provider import AIProvider, AIProviderError, AIProviderTimeoutError


class GroqProvider(AIProvider):
    """Structured extraction through Groq's OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._authorization = f"Bearer {api_key}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
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
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_extraction",
                    "strict": True,
                    "schema": output_schema,
                },
            },
        }
        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": self._authorization},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise AIProviderTimeoutError("Groq request timed out") from exc
        except httpx.HTTPError as exc:
            raise AIProviderError("Groq request failed") from exc

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise AIProviderError("Groq returned an invalid structured response") from exc

        if not isinstance(result, dict):
            raise AIProviderError("Groq structured response must be a JSON object")
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
