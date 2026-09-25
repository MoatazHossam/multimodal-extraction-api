from typing import Any

from fastapi.testclient import TestClient

from app.ai.provider import AIProvider
from app.main import create_app
from app.services.action_detection_service import ActionDetectionService
from app.services.extraction_service import ExtractionService
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.assistance_request import AssistanceRequestWorkflow
from app.workflows.base import WorkflowRegistry


class APIStubProvider(AIProvider):
    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "file_number": "5678",
            "priority": "urgent",
            "request_type": "financial_assistance",
            "requester_name": None,
            "action_needed": "مساعدة في دفع المصروفات الدراسية",
        }


class ActionAPIStubProvider(AIProvider):
    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "actions": [
                {"action_type": "create_meeting", "source_text": "Schedule a meeting"},
                {"action_type": "send_email", "source_text": "email him the details"},
                {"action_type": "create_reminder", "source_text": "remind me before"},
            ]
        }


def test_health() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_text_extraction_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.extraction_service = ExtractionService(
            APIStubProvider(), WorkflowRegistry([AssistanceRequestWorkflow()])
        )
        response = client.post(
            "/api/v1/extract/text",
            json={
                "text": "الملف رقم 5678 يحتاج مساعدة في دفع المصروفات الدراسية والأمر عاجل",
                "workflow": "assistance_request",
            },
        )

    assert response.status_code == 200
    assert response.json()["missing_fields"] == ["requester_name"]
    assert response.json()["source_type"] == "text"


def test_blank_text_is_rejected() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/extract/text", json={"text": "  ", "workflow": "assistance_request"}
        )
    assert response.status_code == 422


def test_action_detection_endpoint_returns_ordered_actions() -> None:
    text = "Schedule a meeting, email him the details, and remind me before."
    app = create_app()
    with TestClient(app) as client:
        app.state.action_detection_service = ActionDetectionService(
            ActionAPIStubProvider(), WorkflowRegistry([ActionDetectionWorkflow()])
        )
        response = client.post("/api/v1/actions/detect", json={"text": text})

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "text": text,
        "actions": [
            {"action_type": "create_meeting", "source_text": "Schedule a meeting"},
            {"action_type": "send_email", "source_text": "email him the details"},
            {"action_type": "create_reminder", "source_text": "remind me before"},
        ],
    }
