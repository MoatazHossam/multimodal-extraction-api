import json
from typing import Any

from app.ai.provider import AIProvider
from app.schemas.requests import ActionParsingRequest
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import ActionParameterExtractionService
from app.services.action_parsing_service import ActionParsingService
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.base import WorkflowRegistry
from app.workflows.email_parameters import EmailParameterWorkflow
from app.workflows.meeting_parameters import MeetingParameterWorkflow
from app.workflows.reminder_parameters import ReminderParameterWorkflow
from app.workflows.task_parameters import TaskParameterWorkflow


class SequentialProvider(AIProvider):
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = iter(results)
        self.calls: list[str] = []

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(user_text)
        return next(self.results)


def make_service(results: list[dict[str, Any]]) -> tuple[ActionParsingService, SequentialProvider]:
    provider = SequentialProvider(results)
    registry = WorkflowRegistry(
        [
            ActionDetectionWorkflow(),
            TaskParameterWorkflow(),
            MeetingParameterWorkflow(),
            EmailParameterWorkflow(),
            ReminderParameterWorkflow(),
        ]
    )
    return (
        ActionParsingService(
            ActionDetectionService(provider, registry),
            ActionParameterExtractionService(provider, registry),
        ),
        provider,
    )


def parse_request(text: str) -> ActionParsingRequest:
    return ActionParsingRequest(
        text=text,
        reference_datetime="2026-09-25T05:50:00+04:00",
        timezone="Asia/Dubai",
    )


async def test_parses_emirati_actions_in_order_without_future_context() -> None:
    text = (
        "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل "
        "وذكرني قبل الاجتماع بساعة"
    )
    service, provider = make_service(
        [
            {
                "actions": [
                    {
                        "action_type": "create_meeting",
                        "source_text": "سو لي اجتماع مع أحمد باچر الساعة عشر",
                    },
                    {"action_type": "send_email", "source_text": "وطرش له إيميل بالتفاصيل"},
                    {
                        "action_type": "create_reminder",
                        "source_text": "وذكرني قبل الاجتماع بساعة",
                    },
                ]
            },
            {
                "title": None,
                "attendees": ["أحمد باچر"],
                "date": None,
                "time": "10:00",
                "duration_minutes": None,
                "location": None,
                "agenda": None,
            },
            {"to": ["أحمد"], "cc": [], "subject": None, "body": None},
            {
                "reminder_text": None,
                "date": None,
                "time": None,
                "relative_to": "الاجتماع",
                "offset_minutes": -60,
            },
        ]
    )

    result = await service.parse(parse_request(text))

    assert [action.action_type for action in result.actions] == [
        "create_meeting",
        "send_email",
        "create_reminder",
    ]
    meeting, email, reminder = result.actions
    assert meeting.parameters.attendees == ["أحمد"]
    assert meeting.parameters.date.isoformat() == "2026-09-26"
    assert meeting.parameters.time.strftime("%H:%M") == "10:00"
    assert meeting.parameters.agenda is None
    assert email.parameters.to == ["أحمد"]
    assert reminder.parameters.relative_to == "الاجتماع"
    assert reminder.parameters.offset_minutes == -60
    meeting_context = json.loads(provider.calls[1])
    assert meeting_context["action_source_text"] == "سو لي اجتماع مع أحمد باچر الساعة عشر"
    assert "إيميل" not in provider.calls[1]
    assert "ذكرني" not in provider.calls[1]


async def test_real_emirati_classification_regression_is_parsed_end_to_end() -> None:
    text = "طرش إيميل لأحمد اطلب منه تقرير عن الحالات اليومية وسو اجتماع باكر الساعة ١٠"
    service, _ = make_service(
        [
            {
                "actions": [
                    {"action_type": "send_email", "source_text": text},
                    {"action_type": "create_task", "source_text": "سو اجتماع باكر الساعة ١٠"},
                ]
            },
            {
                "to": ["أحمد"],
                "cc": [],
                "subject": None,
                "body": "أرسل تقريراً عن الحالات اليومية",
            },
            {
                "title": None,
                "attendees": [],
                "date": None,
                "time": "10:00",
                "duration_minutes": None,
                "location": None,
                "agenda": None,
            },
        ]
    )

    result = await service.parse(parse_request(text))

    assert [action.action_type for action in result.actions] == ["send_email", "create_meeting"]
    assert result.actions[0].parameters.to == ["أحمد"]
    assert "@" not in result.actions[0].parameters.to[0]
    assert "الحالات اليومية" in result.actions[0].parameters.body
    assert result.actions[1].parameters.date.isoformat() == "2026-09-26"


async def test_unsupported_actions_are_returned_without_parameter_generation() -> None:
    text = "اكتب ملاحظة"
    service, provider = make_service(
        [
            {
                "actions": [
                    {"action_type": "create_note", "source_text": text},
                    {"action_type": "follow_up", "source_text": "تابع الموضوع"},
                ]
            }
        ]
    )

    result = await service.parse(parse_request(text))

    assert len(provider.calls) == 1
    assert [action.parameter_status for action in result.actions] == [
        "not_supported",
        "not_supported",
    ]
    assert all(action.parameters is None for action in result.actions)
    assert all(action.missing_fields == [] for action in result.actions)
