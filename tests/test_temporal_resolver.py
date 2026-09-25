from datetime import datetime

import pytest

from app.services.temporal_resolver import TemporalResolution, resolve_temporal

REFERENCE = datetime.fromisoformat("2026-09-25T04:44:00+04:00")


def test_temporal_resolution_can_be_created() -> None:
    assert TemporalResolution() == TemporalResolution(
        date=None,
        relative_to=None,
        offset_minutes=None,
    )


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("باچر", "2026-09-26"),
        ("باجر", "2026-09-26"),
        ("باكر", "2026-09-26"),
        ("عقب باچر", "2026-09-27"),
        ("عقب باجر", "2026-09-27"),
        ("الأحد الياي", "2026-09-27"),
        ("الجمعة الياية", "2026-10-02"),
        ("day after tomorrow", "2026-09-27"),
    ],
)
def test_resolves_relative_dates_in_dubai(phrase: str, expected: str) -> None:
    result = resolve_temporal(phrase, REFERENCE, "Asia/Dubai")
    assert result.date is not None
    assert result.date.isoformat() == expected


def test_converts_reference_to_requested_timezone_before_resolving() -> None:
    reference = datetime.fromisoformat("2026-09-25T22:30:00+00:00")
    result = resolve_temporal("باچر", reference, "Asia/Dubai")
    assert result.date is not None
    assert result.date.isoformat() == "2026-09-27"


@pytest.mark.parametrize(
    ("phrase", "relative_to", "offset"),
    [
        ("قبل الاجتماع بساعة", "الاجتماع", -60),
        ("قبل الموعد بنص ساعة", "الموعد", -30),
        ("عقب الاجتماع بساعة", "الاجتماع", 60),
        ("بعد الاجتماع بساعتين", "الاجتماع", 120),
    ],
)
def test_resolves_safe_relative_reminders(
    phrase: str, relative_to: str, offset: int
) -> None:
    result = resolve_temporal(phrase, REFERENCE, "Asia/Dubai")
    assert result.relative_to == relative_to
    assert result.offset_minutes == offset


def test_does_not_resolve_pronoun_without_antecedent() -> None:
    result = resolve_temporal("ذكرني قبلها بربع ساعة", REFERENCE, "Asia/Dubai")
    assert result.relative_to is None
    assert result.offset_minutes is None
