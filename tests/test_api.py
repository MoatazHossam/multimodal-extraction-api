from typing import Any

from fastapi.testclient import TestClient

from app.ai.provider import AIProvider
from app.main import create_app
from app.schemas.responses import ActionParsingResponse, ParsedUnsupportedAction
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import ActionParameterExtractionService
from app.services.extraction_service import ExtractionService
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.assistance_request import AssistanceRequestWorkflow
from app.workflows.base import WorkflowRegistry
from app.workflows.meeting_parameters import MeetingParameterWorkflow


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


class ParameterAPIStubProvider(AIProvider):
    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "title": None,
            "attendees": ["أحمد"],
            "date": "2026-09-26",
            "time": "10:00",
            "duration_minutes": None,
            "location": None,
            "agenda": None,
        }


class ParsingAPIStubService:
    async def parse(self, request: Any) -> ActionParsingResponse:
        return ActionParsingResponse(
            text=request.text,
            actions=[
                ParsedUnsupportedAction(
                    action_type="create_note", source_text=request.text
                )
            ],
        )


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


def test_action_parameter_extraction_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.action_parameter_service = ActionParameterExtractionService(
            ParameterAPIStubProvider(), WorkflowRegistry([MeetingParameterWorkflow()])
        )
        response = client.post(
            "/api/v1/actions/extract-parameters",
            json={
                "original_text": "اعمل اجتماع مع أحمد بكرة الساعة 10",
                "action": {
                    "action_type": "create_meeting",
                    "source_text": "اعمل اجتماع مع أحمد بكرة الساعة 10",
                },
                "reference_datetime": "2026-09-25T04:45:00+04:00",
                "timezone": "Asia/Dubai",
            },
        )

    assert response.status_code == 200
    assert response.json()["parameters"]["time"] == "10:00"
    assert response.json()["parameters"]["date"] == "2026-09-26"
    assert response.json()["missing_fields"] == []


def test_unsupported_parameter_action_returns_422_without_calling_provider() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/actions/extract-parameters",
            json={
                "original_text": "Nothing actionable",
                "action": {"action_type": "unknown", "source_text": "Nothing actionable"},
                "reference_datetime": "2026-09-25T04:45:00+04:00",
                "timezone": "Asia/Dubai",
            },
        )
    assert response.status_code == 422


def test_action_parsing_endpoint_returns_unsupported_actions_without_failure() -> None:
    app = create_app()
    with TestClient(app) as client:
        app.state.action_parsing_service = ParsingAPIStubService()
        response = client.post(
            "/api/v1/actions/parse",
            json={
                "text": "اكتب ملاحظة",
                "reference_datetime": "2026-09-25T05:50:00+04:00",
                "timezone": "Asia/Dubai",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "text": "اكتب ملاحظة",
        "actions": [
            {
                "action_type": "create_note",
                "source_text": "اكتب ملاحظة",
                "parameters": None,
                "missing_fields": [],
                "parameter_status": "not_supported",
            }
        ],
    }


def test_action_parsing_request_reuses_temporal_validation() -> None:
    with TestClient(create_app()) as client:
        naive_datetime = client.post(
            "/api/v1/actions/parse",
            json={
                "text": "ذكرني",
                "reference_datetime": "2026-09-25T05:50:00",
                "timezone": "Asia/Dubai",
            },
        )
        invalid_timezone = client.post(
            "/api/v1/actions/parse",
            json={
                "text": "ذكرني",
                "reference_datetime": "2026-09-25T05:50:00+04:00",
                "timezone": "Dubai",
            },
        )

    assert naive_datetime.status_code == 422
    assert invalid_timezone.status_code == 422
