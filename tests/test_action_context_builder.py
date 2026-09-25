from app.services.action_context_builder import build_action_context

ORIGINAL = "سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل وذكرني قبل الاجتماع بساعة"


def test_builds_prefix_context_for_each_exact_action() -> None:
    meeting = build_action_context(ORIGINAL, "سو لي اجتماع مع أحمد باچر الساعة عشر")
    email = build_action_context(ORIGINAL, "وطرش له إيميل بالتفاصيل")
    reminder = build_action_context(ORIGINAL, "وذكرني قبل الاجتماع بساعة")

    assert meeting.exact_match is True
    assert meeting.prior_context == ""
    assert meeting.contextual_text == "سو لي اجتماع مع أحمد باچر الساعة عشر"
    assert email.prior_context == "سو لي اجتماع مع أحمد باچر الساعة عشر "
    assert email.contextual_text == ("سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل")
    assert reminder.prior_context.endswith("وطرش له إيميل بالتفاصيل ")
    assert reminder.contextual_text == ORIGINAL


def test_unmatched_action_uses_conservative_full_text_fallback() -> None:
    context = build_action_context(ORIGINAL, "طرش له ايميل بالتفاصيل")

    assert context.exact_match is False
    assert context.prior_context == ORIGINAL
    assert context.contextual_text == ORIGINAL
    assert context.action_source_text == "طرش له ايميل بالتفاصيل"
