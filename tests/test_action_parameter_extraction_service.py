import json
from typing import Any

import pytest

from app.ai.provider import AIProvider
from app.schemas.requests import ActionParameterExtractionRequest
from app.services.action_parameter_extraction_service import (
    ActionParameterExtractionService,
    ActionParameterOutputError,
    UnsupportedActionTypeError,
)
from app.workflows.base import WorkflowRegistry
from app.workflows.email_parameters import EmailParameterWorkflow
from app.workflows.meeting_parameters import MeetingParameterWorkflow
from app.workflows.reminder_parameters import ReminderParameterWorkflow
from app.workflows.task_parameters import TaskParameterWorkflow


class StubProvider(AIProvider):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.user_text: str | None = None
        self.system_prompt: str | None = None
        self.output_schema: dict[str, Any] | None = None

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.system_prompt = system_prompt
        self.user_text = user_text
        self.output_schema = output_schema
        return self.result


def make_service(result: dict[str, Any]) -> tuple[ActionParameterExtractionService, StubProvider]:
    provider = StubProvider(result)
    registry = WorkflowRegistry(
        [
            TaskParameterWorkflow(),
            MeetingParameterWorkflow(),
            EmailParameterWorkflow(),
            ReminderParameterWorkflow(),
        ]
    )
    return ActionParameterExtractionService(provider, registry), provider


def request_for(
    action_type: str, source_text: str, *, original_text: str | None = None
) -> ActionParameterExtractionRequest:
    return ActionParameterExtractionRequest.model_validate(
        {
            "original_text": original_text or source_text,
            "action": {"action_type": action_type, "source_text": source_text},
            "reference_datetime": "2026-09-25T04:45:00+04:00",
            "timezone": "Asia/Dubai",
        }
    )


@pytest.mark.parametrize(
    ("source_text", "output", "expected_title"),
    [
        (
            "اعمل مهمة لأحمد يراجع التقرير بكرة وخليها عاجلة",
            {
                "title": "مراجعة التقرير",
                "assignees": ["أحمد"],
                "due_date": "2026-09-26",
                "due_time": None,
                "priority": "urgent",
                "description": "مراجعة التقرير",
            },
            "مراجعة التقرير",
        ),
        (
            "Create a task for Ahmed to review the report tomorrow",
            {
                "title": "Review the report",
                "assignees": ["Ahmed"],
                "due_date": "2026-09-26",
                "due_time": None,
                "priority": None,
                "description": "Review the report",
            },
            "Review the report",
        ),
    ],
)
async def test_extracts_arabic_and_english_tasks(
    source_text: str, output: dict[str, Any], expected_title: str
) -> None:
    service, provider = make_service(output)

    result = await service.extract(request_for("create_task", source_text))

    assert result.parameters.title == expected_title
    assert result.missing_fields == []
    context = json.loads(provider.user_text or "")
    assert context["reference_datetime"] == "2026-09-25T04:45:00+04:00"
    assert context["timezone"] == "Asia/Dubai"


@pytest.mark.parametrize(
    ("source_text", "output", "attendee"),
    [
        (
            "اعمل اجتماع مع أحمد بكرة الساعة 10 لمناقشة الميزانية",
            {
                "title": "مناقشة الميزانية",
                "attendees": ["أحمد"],
                "date": "2026-09-26",
                "time": "10:00",
                "duration_minutes": None,
                "location": None,
                "agenda": "مناقشة الميزانية",
            },
            "أحمد",
        ),
        (
            "Schedule a meeting with Ahmed tomorrow at 10 to discuss the budget",
            {
                "title": "Discuss the budget",
                "attendees": ["Ahmed"],
                "date": "2026-09-26",
                "time": "10:00",
                "duration_minutes": None,
                "location": None,
                "agenda": "Discuss the budget",
            },
            "Ahmed",
        ),
    ],
)
async def test_extracts_meetings_and_resolves_tomorrow(
    source_text: str, output: dict[str, Any], attendee: str
) -> None:
    service, _ = make_service(output)

    result = await service.extract(request_for("create_meeting", source_text))

    assert result.parameters.attendees == [attendee]
    assert result.parameters.date.isoformat() == "2026-09-26"
    assert result.parameters.time.strftime("%H:%M") == "10:00"
    assert result.missing_fields == []


async def test_resolves_arabic_email_pronoun_without_inventing_address() -> None:
    original = "اعمل اجتماع مع أحمد بكرة الساعة 10 وابعتله إيميل بالتفاصيل"
    service, provider = make_service(
        {"to": ["أحمد"], "cc": [], "subject": "تفاصيل الاجتماع", "body": "تفاصيل الاجتماع"}
    )

    result = await service.extract(
        request_for("send_email", "ابعتله إيميل بالتفاصيل", original_text=original)
    )

    assert result.parameters.to == ["أحمد"]
    assert "@" not in result.parameters.to[0]
    assert json.loads(provider.user_text or "")["original_text"] == original


async def test_extracts_reminder_relative_to_meeting() -> None:
    service, _ = make_service(
        {
            "reminder_text": "تذكير بالاجتماع",
            "date": None,
            "time": None,
            "relative_to": "الاجتماع",
            "offset_minutes": -60,
        }
    )

    result = await service.extract(
        request_for("create_reminder", "حطلي تذكير قبل الاجتماع بساعة")
    )

    assert result.parameters.offset_minutes == -60
    assert result.missing_fields == []


async def test_reports_only_execution_critical_missing_fields() -> None:
    service, _ = make_service(
        {
            "title": None,
            "assignees": [],
            "due_date": None,
            "due_time": None,
            "priority": None,
            "description": None,
        }
    )
    result = await service.extract(request_for("create_task", "اعمل مهمة"))
    assert result.missing_fields == ["title"]


@pytest.mark.parametrize(
    "malformed_output",
    [
        {},
        {"to": ["Ahmed"], "cc": [], "subject": None},
        {"to": ["Ahmed"], "cc": [], "subject": None, "body": None, "extra": True},
    ],
)
async def test_rejects_malformed_provider_output(malformed_output: dict[str, Any]) -> None:
    service, _ = make_service(malformed_output)
    with pytest.raises(ActionParameterOutputError):
        await service.extract(request_for("send_email", "Email Ahmed"))


async def test_rejects_unsupported_action_type() -> None:
    service, _ = make_service({})
    with pytest.raises(UnsupportedActionTypeError):
        await service.extract(request_for("unknown", "Maybe do something"))
