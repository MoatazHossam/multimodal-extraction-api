import json
import logging

import httpx
import pytest

from app.ai.groq_provider import GroqProvider
from app.ai.provider import AIProviderError, AIProviderTimeoutError


def make_provider(handler: httpx.MockTransport) -> tuple[GroqProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=handler, base_url="https://groq.test/openai/v1")
    return (
        GroqProvider(
            api_key="secret-test-key",
            base_url="https://unused.test",
            model="test-model",
            timeout_seconds=1,
            client=client,
        ),
        client,
    )


async def test_parses_groq_structured_response() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        assert body["model"] == "test-model"
        assert body["reasoning_effort"] == "none"
        assert body["messages"] == [
            {"role": "system", "content": "extract"},
            {"role": "user", "content": "input"},
        ]
        assert body["response_format"]["json_schema"]["schema"] == schema
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"value":"ok"}'}}]})

    provider, client = make_provider(httpx.MockTransport(handler))
    result = await provider.extract_structured(
        system_prompt="extract", user_text="input", output_schema=schema
    )

    assert result == {"value": "ok"}
    await client.aclose()


async def test_maps_groq_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider, client = make_provider(httpx.MockTransport(handler))
    with pytest.raises(AIProviderTimeoutError, match="Groq request timed out"):
        await provider.extract_structured(
            system_prompt="extract", user_text="input", output_schema={"type": "object"}
        )
    await client.aclose()


async def test_maps_groq_http_failure() -> None:
    provider, client = make_provider(
        httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "limited"}))
    )
    with pytest.raises(AIProviderError, match="Groq request failed"):
        await provider.extract_structured(
            system_prompt="extract", user_text="input", output_schema={"type": "object"}
        )
    await client.aclose()


@pytest.mark.parametrize(
    "content",
    ["not json", '```json\n{"value": "ok"}\n```', "[]"],
)
async def test_rejects_malformed_or_non_object_groq_output(content: str) -> None:
    provider, client = make_provider(
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )
        )
    )
    with pytest.raises(AIProviderError):
        await provider.extract_structured(
            system_prompt="extract", user_text="input", output_schema={"type": "object"}
        )
    await client.aclose()


async def test_api_key_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        base_url="https://groq.test/openai/v1",
    )
    provider = GroqProvider(
        api_key="never-log-this-key",
        base_url="https://groq.test/openai/v1",
        model="test-model",
        timeout_seconds=1,
        client=client,
    )

    with pytest.raises(AIProviderError):
        await provider.extract_structured(
            system_prompt="extract", user_text="input", output_schema={"type": "object"}
        )

    assert "never-log-this-key" not in caplog.text
    await client.aclose()
