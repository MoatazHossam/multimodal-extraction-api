from app.ai.groq_provider import GroqProvider
from app.ai.ollama_provider import OllamaProvider
from app.ai.provider import AIProvider
from app.config import Settings


def create_ai_provider(settings: Settings) -> AIProvider:
    """Build the single provider selected by application configuration."""
    if settings.ai_provider == "ollama":
        return OllamaProvider(
            base_url=str(settings.ollama_base_url),
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    if settings.ai_provider == "groq":
        # Settings validation guarantees the key exists for this branch.
        assert settings.groq_api_key is not None
        return GroqProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=str(settings.groq_base_url),
            model=settings.groq_model,
            timeout_seconds=settings.groq_timeout_seconds,
        )
    raise ValueError(f"Unsupported AI provider: {settings.ai_provider}")
