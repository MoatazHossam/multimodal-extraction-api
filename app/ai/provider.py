from abc import ABC, abstractmethod
from typing import Any


class AIProviderError(RuntimeError):
    """Base exception for failures returned by an AI provider."""


class AIProviderTimeoutError(AIProviderError):
    """Raised when the AI provider does not respond before the timeout."""


class AIProvider(ABC):
    """Provider-neutral interface for structured extraction."""

    @abstractmethod
    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract a JSON-compatible object matching ``output_schema``."""

    async def close(self) -> None:
        """Release provider resources, if any."""
        return None
