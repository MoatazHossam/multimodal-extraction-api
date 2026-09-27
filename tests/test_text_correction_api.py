from typing import Any

from fastapi.testclient import TestClient

from app.ai.provider import AIProvider, AIProviderError, AIProviderTimeoutError
from app.main import create_app
from app.services.text_correction_service import TextCorrectionService
from app.workflows.base import WorkflowRegistry
from app.workflows.text_correction import TextCorrectionWorkflow


class ResultProvider(AIProvider):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return self.result


class FailingProvider(AIProvider):
    def __init__(self, error: AIProviderError) -> None:
        self.error = error

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        raise self.error


def service(provider: AIProvider) -> TextCorrectionService:
    return TextCorrectionService(
        provider,
        WorkflowRegistry(
            [
                TextCorrectionWorkflow("formal"),
                TextCorrectionWorkflow("asr_repair"),
                TextCorrectionWorkflow("asr_formal"),
            ]
        ),
    )


def test_text_correction_endpoint_returns_strict_response() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.text_correction_service = service(
            ResultProvider({"corrected_text": "واحد اثنين ثلاثة"})
        )
        response = client.post(
            "/api/v1/text/correct",
            json={"text": "وحد اتنينتالتة", "mode": "asr_formal"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "original_text": "وحد اتنينتالتة",
        "corrected_text": "واحد اثنين ثلاثة",
        "changed": True,
        "mode": "asr_formal",
    }


def test_empty_text_is_rejected() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/text/correct", json={"text": "  ", "mode": "formal"}
        )

    assert response.status_code == 422


def test_unsupported_mode_is_rejected() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/text/correct", json={"text": "نص", "mode": "creative"}
        )

    assert response.status_code == 422


def test_provider_timeout_maps_to_504() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.text_correction_service = service(
            FailingProvider(AIProviderTimeoutError("provider timed out"))
        )
        response = client.post(
            "/api/v1/text/correct", json={"text": "نص", "mode": "formal"}
        )

    assert response.status_code == 504
    assert response.json() == {"detail": "provider timed out"}


def test_provider_failure_maps_to_502() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.text_correction_service = service(
            FailingProvider(AIProviderError("provider failed"))
        )
        response = client.post(
            "/api/v1/text/correct", json={"text": "نص", "mode": "formal"}
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "AI text correction service failed"}
