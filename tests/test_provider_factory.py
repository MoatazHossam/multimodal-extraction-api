import pytest
from pydantic import ValidationError

from app.ai.factory import create_ai_provider
from app.ai.groq_provider import GroqProvider
from app.ai.ollama_provider import OllamaProvider
from app.config import Settings


async def test_factory_selects_ollama_by_default() -> None:
    provider = create_ai_provider(Settings(_env_file=None))

    assert isinstance(provider, OllamaProvider)
    await provider.close()


async def test_factory_selects_groq() -> None:
    settings = Settings(
        _env_file=None,
        ai_provider="groq",
        groq_api_key="test-key",
    )

    provider = create_ai_provider(settings)

    assert isinstance(provider, GroqProvider)
    await provider.close()


def test_unsupported_provider_is_rejected_during_config_validation() -> None:
    with pytest.raises(ValidationError, match="ai_provider"):
        Settings(_env_file=None, ai_provider="unsupported")


def test_groq_requires_api_key_during_config_validation() -> None:
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(_env_file=None, ai_provider="groq")
