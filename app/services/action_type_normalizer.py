import re
from collections.abc import Sequence

from app.schemas.responses import DetectedAction

# Treat digits and punctuation as valid token boundaries. This covers both Arabic-Indic
# timestamps and ordinary punctuation without matching meeting nouns inside another word.
_EXPLICIT_MEETING_NOUN = re.compile(
    r"(?<![^\W\d])(?:اجتماع|meeting)(?![^\W\d])",
    re.IGNORECASE,
)


def normalize_action_types(actions: Sequence[DetectedAction]) -> list[DetectedAction]:
    """Correct only high-confidence action-type contradictions.

    Explicit meeting nouns outweigh a model's generic-task classification. All other
    classifications and every source snippet remain exactly as supplied.
    """
    return [
        action.model_copy(update={"action_type": "create_meeting"})
        if action.action_type == "create_task" and _EXPLICIT_MEETING_NOUN.search(action.source_text)
        else action
        for action in actions
    ]
