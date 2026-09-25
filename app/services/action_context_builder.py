from dataclasses import dataclass


@dataclass(frozen=True)
class ActionContext:
    """Text available while extracting parameters for one detected action."""

    prior_context: str
    action_source_text: str
    contextual_text: str
    exact_match: bool


def build_action_context(original_text: str, action_source_text: str) -> ActionContext:
    """Isolate an action from any actions that follow it in the original text.

    Detection returns source text copied from the input in the normal case.  When that
    invariant does not hold, retaining the full original text is safer than guessing at
    clause boundaries and preserves the previous parameter-extraction behavior.
    """

    action_start = original_text.find(action_source_text)
    if action_start == -1:
        return ActionContext(
            prior_context=original_text,
            action_source_text=action_source_text,
            contextual_text=original_text,
            exact_match=False,
        )

    action_end = action_start + len(action_source_text)
    return ActionContext(
        prior_context=original_text[:action_start],
        action_source_text=action_source_text,
        contextual_text=original_text[:action_end],
        exact_match=True,
    )
