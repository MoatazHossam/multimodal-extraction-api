import pytest

from app.schemas.responses import ActionType, DetectedAction
from app.services.action_type_normalizer import normalize_action_types


@pytest.mark.parametrize(
    ("source_text", "model_type", "expected_type"),
    [
        ("سو اجتماع باكر الساعة ١٠", "create_task", "create_meeting"),
        ("حط اجتماع ويا أحمد", "create_task", "create_meeting"),
        ("رتب اجتماع مع سالم", "create_task", "create_meeting"),
        ("Schedule a meeting tomorrow", "create_task", "create_meeting"),
        ("سو مهمة لأحمد", "create_task", "create_task"),
        ("حط مهمة لسالم يراجع التقرير", "create_task", "create_task"),
        ("سو اجتماع، باكر الساعة ١٠", "create_task", "create_meeting"),
        ("١٠اجتماع١١", "create_task", "create_meeting"),
        ("meeting_notes", "create_task", "create_task"),
        ("سو اجتماع باكر", "create_meeting", "create_meeting"),
        ("ناقش اجتماع باكر", "send_email", "send_email"),
    ],
)
def test_normalizes_only_task_classifications_with_explicit_meeting_nouns(
    source_text: str, model_type: ActionType, expected_type: ActionType
) -> None:
    action = DetectedAction(action_type=model_type, source_text=source_text)

    result = normalize_action_types([action])

    assert result[0].action_type == expected_type
    assert result[0].source_text == source_text


def test_preserves_action_order_and_count() -> None:
    actions = [
        DetectedAction(action_type="send_email", source_text="طرش إيميل لأحمد"),
        DetectedAction(action_type="create_task", source_text="سو اجتماع باكر الساعة ١٠"),
    ]

    result = normalize_action_types(actions)

    assert len(result) == len(actions)
    assert [action.action_type for action in result] == ["send_email", "create_meeting"]
    assert [action.source_text for action in result] == [action.source_text for action in actions]
