import re
from collections.abc import Sequence

from app.schemas.responses import DetectedAction

_TRAILING_PUNCTUATION = re.compile(r"[,،;؛]+$")
_TRAILING_ENGLISH_CONNECTOR = re.compile(r"(?:^|\s)(?:and|then)$", re.IGNORECASE)


def normalize_action_source_texts(
    original_text: str, actions: Sequence[DetectedAction]
) -> list[DetectedAction]:
    """Shorten source texts that overlap a subsequent action's exact source snippet.

    The function deliberately requires every model-provided snippet to occur in the
    original text, in action order. If that evidence is unavailable, it returns the
    validated actions unchanged rather than attempting fuzzy matching or rewriting.
    """
    unchanged = list(actions)
    if len(actions) < 2:
        return unchanged

    spans: list[tuple[int, int]] = []
    search_from = 0
    for action in actions:
        start = original_text.find(action.source_text, search_from)
        if start < 0:
            return unchanged
        spans.append((start, start + len(action.source_text)))
        # Search after the start rather than the end: an oversized action can contain
        # all of the later snippets that we need in order to repair it.
        search_from = start + 1

    normalized = unchanged.copy()
    for index, ((start, end), (next_start, _)) in enumerate(zip(spans, spans[1:], strict=False)):
        if end <= next_start:
            continue

        source_text = _strip_action_boundary(original_text[start:next_start])
        if not source_text:
            return unchanged
        normalized[index] = actions[index].model_copy(update={"source_text": source_text})

    return normalized


def _strip_action_boundary(text: str) -> str:
    """Remove only recognizable separators immediately before the next action."""
    result = text.rstrip()
    while result:
        previous = result
        result = _TRAILING_ENGLISH_CONNECTOR.sub("", result).rstrip()

        if result.endswith("و") and len(result) > 1:
            preceding = result[-2]
            if preceding.isspace() or preceding in ",،;؛":
                result = result[:-1].rstrip()

        result = _TRAILING_PUNCTUATION.sub("", result).rstrip()
        if result == previous:
            break

    return result
