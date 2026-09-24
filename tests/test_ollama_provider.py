import json

import httpx
import pytest

from app.ai.ollama_provider import OllamaProvider
from app.ai.provider import AIProviderError


async def test_calls_ollama_chat_with_schema() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "test-model"
        assert body["format"] == {"type": "object"}
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"content": '{"value": "ok"}'}})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama:11434"
    )
    provider = OllamaProvider(
        base_url="http://unused", model="test-model", timeout_seconds=1, client=client
    )

    result = await provider.extract_structured(
        system_prompt="extract", user_text="input", output_schema={"type": "object"}
    )

    assert result == {"value": "ok"}
    await client.aclose()


async def test_rejects_non_json_content() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"message": {"content": "not json"}})
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama:11434")
    provider = OllamaProvider(
        base_url="http://unused", model="test-model", timeout_seconds=1, client=client
    )

    with pytest.raises(AIProviderError):
        await provider.extract_structured(
            system_prompt="extract", user_text="input", output_schema={"type": "object"}
        )
    await client.aclose()

