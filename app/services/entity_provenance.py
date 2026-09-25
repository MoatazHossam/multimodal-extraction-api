import re
import unicodedata
from collections.abc import Collection

MEETING_PARTICIPANT_REFERENCES = frozenset(
    {"وياه", "معاه", "معه", "وياها", "معاها", "معها", "وياهم", "معاهم", "معهم"}
)

TASK_ASSIGNEE_REFERENCES = frozenset({"له", "لها", "لهم", "إله", "إلها", "إلهم"})


def filter_entities_by_provenance(
    entities: list[str],
    *,
    action_source_text: str,
    prior_context: str,
    participant_references: Collection[str],
) -> list[str]:
    """Keep entities explicitly named or explicitly referenced by the current action.

    Reference syntax only authorizes resolution of an entity that is actually present in
    prior context. This prevents a reference from becoming permission to invent a person.
    """
    has_reference = any(
        _contains_term(action_source_text, reference) for reference in participant_references
    )
    return [
        entity
        for entity in entities
        if _contains_term(action_source_text, entity)
        or (has_reference and _contains_term(prior_context, entity))
    ]


def _contains_term(text: str, term: str) -> bool:
    normalized_text = _normalize(text)
    normalized_term = _normalize(term)
    if not normalized_term:
        return False
    # Arabic conjunctions and prepositions are commonly attached to a name (for example,
    # "لأحمد"). Treat that morphology as an exact mention without enabling fuzzy matching.
    optional_arabic_prefix = "[وفبكل]?" if _starts_with_arabic_letter(normalized_term) else ""
    pattern = rf"(?<!\w){optional_arabic_prefix}{re.escape(normalized_term)}(?!\w)"
    return re.search(pattern, normalized_text) is not None


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _starts_with_arabic_letter(value: str) -> bool:
    return bool(value) and "ARABIC" in unicodedata.name(value[0], "")
