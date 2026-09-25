from typing import Any

import pytest

from app.ai.provider import AIProvider
from app.schemas.requests import ActionDetectionRequest
from app.services.action_detection_service import (
    ActionDetectionOutputError,
    ActionDetectionService,
)
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.base import WorkflowRegistry


class StubProvider(AIProvider):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.received_schema: dict[str, Any] | None = None
        self.received_prompt: str | None = None

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.received_prompt = system_prompt
        self.received_schema = output_schema
        return self.result


def make_service(result: dict[str, Any]) -> tuple[ActionDetectionService, StubProvider]:
    provider = StubProvider(result)
    registry = WorkflowRegistry([ActionDetectionWorkflow()])
    return ActionDetectionService(provider, registry), provider


@pytest.mark.parametrize(
    ("text", "action_type"),
    [
        ("Create a task for Ahmed to review the report tomorrow", "create_task"),
        ("اعمل اجتماع مع أحمد بكرة الساعة 10", "create_meeting"),
        ("Email Ahmed the report", "send_email"),
        ("ذكرني بموعد الطبيب غداً", "create_reminder"),
        ("Ahmed was in the office yesterday", "unknown"),
    ],
)
async def test_detects_single_action(text: str, action_type: str) -> None:
    service, provider = make_service(
        {"actions": [{"action_type": action_type, "source_text": text}]}
    )

    result = await service.detect(ActionDetectionRequest(text=text))

    assert result.text == text
    assert result.actions[0].action_type == action_type
    assert result.actions[0].source_text == text
    assert provider.received_schema is not None
    assert provider.received_schema["properties"]["actions"]["minItems"] == 1
    assert provider.received_prompt is not None


@pytest.mark.parametrize(
    ("text", "actions"),
    [
        (
            "Schedule a meeting with Ahmed tomorrow at 10, email him the details, "
            "and remind me one hour before.",
            [
                {
                    "action_type": "create_meeting",
                    "source_text": "Schedule a meeting with Ahmed tomorrow at 10",
                },
                {"action_type": "send_email", "source_text": "email him the details"},
                {
                    "action_type": "create_reminder",
                    "source_text": "remind me one hour before",
                },
            ],
        ),
        (
            "اعمل اجتماع مع أحمد بكرة الساعة 10 وابعتله إيميل بالتفاصيل "
            "وحطلي تذكير قبل الاجتماع بساعة",
            [
                {
                    "action_type": "create_meeting",
                    "source_text": "اعمل اجتماع مع أحمد بكرة الساعة 10",
                },
                {"action_type": "send_email", "source_text": "ابعتله إيميل بالتفاصيل"},
                {
                    "action_type": "create_reminder",
                    "source_text": "حطلي تذكير قبل الاجتماع بساعة",
                },
            ],
        ),
    ],
)
async def test_preserves_order_and_cleanly_segments_multiple_actions(
    text: str, actions: list[dict[str, str]]
) -> None:
    service, _ = make_service(
        {"actions": actions}
    )

    result = await service.detect(ActionDetectionRequest(text=text))

    assert [action.action_type for action in result.actions] == [
        "create_meeting",
        "send_email",
        "create_reminder",
    ]
    assert [action.source_text for action in result.actions] == [
        action["source_text"] for action in actions
    ]


def test_prompt_requires_minimal_action_only_source_text_with_full_context_later() -> None:
    prompt = ActionDetectionWorkflow().system_prompt

    assert "smallest complete clause" in prompt
    assert "Do not include words or clauses belonging to another detected action" in prompt
    assert "Preserve the original language" in prompt
    assert "do not paraphrase unnecessarily or invent missing context" in prompt
    assert "Pronouns and references may remain unresolved" in prompt
    assert "original full text as context" in prompt


@pytest.mark.parametrize(
    "provider_output",
    [
        {},
        {"actions": []},
        {"actions": [{"action_type": "make_phone_call", "source_text": "Call Ahmed"}]},
        {"actions": [{"action_type": "create_task", "source_text": " "}]},
        {"actions": [{"action_type": "create_task", "source_text": "Do it", "extra": 1}]},
    ],
)
async def test_rejects_malformed_provider_output(provider_output: dict[str, Any]) -> None:
    service, _ = make_service(provider_output)

    with pytest.raises(ActionDetectionOutputError):
        await service.detect(ActionDetectionRequest(text="Call Ahmed"))
