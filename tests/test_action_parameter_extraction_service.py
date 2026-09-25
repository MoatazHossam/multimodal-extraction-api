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


async def test_emirati_date_overrides_model_and_is_removed_from_attendee() -> None:
    service, _ = make_service(
        {
            "title": None,
            "attendees": ["أحمد باچر"],
            "date": "2026-09-25",
            "time": "10:00",
            "duration_minutes": None,
            "location": None,
            "agenda": None,
        }
    )
    result = await service.extract(
        request_for("create_meeting", "سو لي اجتماع مع أحمد باچر الساعة 10")
    )
    assert result.parameters.attendees == ["أحمد"]
    assert result.parameters.date.isoformat() == "2026-09-26"
    assert result.parameters.time.strftime("%H:%M") == "10:00"


@pytest.mark.parametrize(
    ("source_text", "attendee"),
    [
        ("سو اجتماع مع أحمد باچر", "أحمد"),
        ("رتب اجتماع ويا سالم", "سالم"),
        ("حط اجتماع بيني وبين محمد", "محمد"),
    ],
)
async def test_retains_attendee_named_in_current_meeting(
    source_text: str, attendee: str
) -> None:
    service, _ = make_service(
        {
            "title": None,
            "attendees": [attendee],
            "date": None,
            "time": None,
            "duration_minutes": None,
            "location": None,
            "agenda": None,
        }
    )

    result = await service.extract(request_for("create_meeting", source_text))

    assert result.parameters.attendees == [attendee]


async def test_meeting_reference_allows_attendee_from_prior_context() -> None:
    original = "كلم أحمد وبعدها سو اجتماع وياه باچر"
    service, _ = make_service(
        {
            "title": None,
            "attendees": ["أحمد"],
            "date": None,
            "time": None,
            "duration_minutes": None,
            "location": None,
            "agenda": None,
        }
    )

    result = await service.extract(
        request_for("create_meeting", "سو اجتماع وياه باچر", original_text=original)
    )

    assert result.parameters.attendees == ["أحمد"]


async def test_task_drops_assignee_found_only_in_prior_context() -> None:
    original = "طرش لأحمد إيميل وسو مهمة لمراجعة التقرير"
    service, _ = make_service(
        {
            "title": "مراجعة التقرير",
            "assignees": ["أحمد"],
            "due_date": None,
            "due_time": None,
            "priority": None,
            "description": "مراجعة التقرير",
        }
    )

    result = await service.extract(
        request_for("create_task", "سو مهمة لمراجعة التقرير", original_text=original)
    )

    assert result.parameters.assignees == []


async def test_task_reference_allows_assignee_from_prior_context() -> None:
    original = "كلم أحمد وبعدها سو له مهمة لمراجعة التقرير"
    service, _ = make_service(
        {
            "title": "مراجعة التقرير",
            "assignees": ["أحمد"],
            "due_date": None,
            "due_time": None,
            "priority": None,
            "description": "مراجعة التقرير",
        }
    )

    result = await service.extract(
        request_for("create_task", "سو له مهمة لمراجعة التقرير", original_text=original)
    )

    assert result.parameters.assignees == ["أحمد"]


async def test_emirati_task_date_and_assignee_cleanup() -> None:
    service, _ = make_service(
        {
            "title": "مراجعة التقرير",
            "assignees": ["أحمد باجر"],
            "due_date": None,
            "due_time": None,
            "priority": None,
            "description": None,
        }
    )
    result = await service.extract(request_for("create_task", "كلف أحمد يراجع التقرير باجر"))
    assert result.parameters.assignees == ["أحمد"]
    assert result.parameters.due_date.isoformat() == "2026-09-26"


@pytest.mark.parametrize("person", ["باكر محمد", "باجر علي"])
async def test_does_not_remove_leading_temporal_name_token(person: str) -> None:
    service, _ = make_service(
        {
            "title": "مهمة",
            "assignees": [person],
            "due_date": None,
            "due_time": None,
            "priority": None,
            "description": None,
        }
    )
    result = await service.extract(request_for("create_task", f"كلف {person}"))
    assert result.parameters.assignees == [person]


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
    context = json.loads(provider.user_text or "")
    assert context["prior_context"] == "اعمل اجتماع مع أحمد بكرة الساعة 10 و"
    assert context["action_source_text"] == "ابعتله إيميل بالتفاصيل"


async def test_emirati_action_context_isolates_future_actions() -> None:
    original = (
        "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل وذكرني قبل الاجتماع بساعة"
    )
    service, provider = make_service(
        {
            "title": None,
            "attendees": ["أحمد"],
            "date": "2026-09-26",
            "time": "10:00",
            "duration_minutes": None,
            "location": None,
            "agenda": None,
        }
    )

    result = await service.extract(
        request_for(
            "create_meeting",
            "سو لي اجتماع مع أحمد باچر الساعة عشر",
            original_text=original,
        )
    )

    context = json.loads(provider.user_text or "")
    assert context == {
        "prior_context": "",
        "action_source_text": "سو لي اجتماع مع أحمد باچر الساعة عشر",
        "reference_datetime": "2026-09-25T04:45:00+04:00",
        "timezone": "Asia/Dubai",
    }
    serialized_context = provider.user_text or ""
    assert "طرش له إيميل بالتفاصيل" not in serialized_context
    assert "ذكرني قبل الاجتماع بساعة" not in serialized_context
    assert result.parameters.title is None
    assert result.parameters.agenda is None
    assert "Email, reminder, and task instructions" in (provider.system_prompt or "")


async def test_emirati_email_keeps_meeting_context_for_pronoun_resolution() -> None:
    original = (
        "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل وذكرني قبل الاجتماع بساعة"
    )
    service, provider = make_service(
        {"to": ["أحمد"], "cc": [], "subject": None, "body": "تفاصيل الاجتماع"}
    )

    result = await service.extract(
        request_for("send_email", "وطرش له إيميل بالتفاصيل", original_text=original)
    )

    context = json.loads(provider.user_text or "")
    assert "اجتماع مع أحمد" in context["prior_context"]
    assert context["action_source_text"] == "وطرش له إيميل بالتفاصيل"
    assert "ذكرني قبل الاجتماع بساعة" not in (provider.user_text or "")
    assert result.parameters.to == ["أحمد"]


async def test_emirati_reminder_keeps_preceding_action_context() -> None:
    original = (
        "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل وذكرني قبل الاجتماع بساعة"
    )
    service, provider = make_service(
        {
            "reminder_text": "تذكير بالاجتماع",
            "date": None,
            "time": None,
            "relative_to": "الاجتماع",
            "offset_minutes": -60,
        }
    )

    result = await service.extract(
        request_for("create_reminder", "وذكرني قبل الاجتماع بساعة", original_text=original)
    )

    context = json.loads(provider.user_text or "")
    assert "اجتماع مع أحمد" in context["prior_context"]
    assert "وطرش له إيميل بالتفاصيل" in context["prior_context"]
    assert context["action_source_text"] == "وذكرني قبل الاجتماع بساعة"
    assert result.parameters.relative_to == "الاجتماع"
    assert result.parameters.offset_minutes == -60


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

    result = await service.extract(request_for("create_reminder", "حطلي تذكير قبل الاجتماع بساعة"))

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
