import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TemporalResolution:
    """Only temporal facts that can be derived without a model or a clock."""

    date: date | None = None
    relative_to: str | None = None
    offset_minutes: int | None = None


_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")


def _normalize(text: str) -> str:
    # This representation is for matching only; callers retain the original text.
    return _DIACRITICS.sub("", text).translate(str.maketrans("أإآٱچ", "ااااج"))


_DAY_AFTER = (
    "عقب باجر",
    "عقب باكر",
    "بعد باجر",
    "بعد بكرة",
    "day after tomorrow",
)
_TOMORROW = ("باجر", "باكر", "بكرة", "غدا", "tomorrow")
_TODAY = ("اليوم", "هاليوم", "today", "الليلة")

_WEEKDAYS = {
    "الاثنين": 0,
    "الثلاثاء": 1,
    "الاربعاء": 2,
    "الخميس": 3,
    "الجمعة": 4,
    "السبت": 5,
    "الاحد": 6,
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _contains(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def resolve_temporal(
    source_text: str, reference_datetime: datetime, timezone: str
) -> TemporalResolution:
    """Resolve a deliberately small, deterministic UAE scheduling vocabulary."""
    local_date = reference_datetime.astimezone(ZoneInfo(timezone)).date()
    text = _normalize(source_text)

    # Longer phrases must win over their constituent tomorrow token.
    if any(_contains(text, phrase) for phrase in _DAY_AFTER):
        resolved_date = local_date + timedelta(days=2)
    elif _contains(text, "عقب ثلاث ايام"):
        resolved_date = local_date + timedelta(days=3)
    elif _contains(text, "عقب يومين"):
        resolved_date = local_date + timedelta(days=2)
    elif _contains(text, "عقب يوم"):
        resolved_date = local_date + timedelta(days=1)
    elif _contains(text, "عقب اسبوع") or _contains(text, "الاسبوع الياي"):
        resolved_date = local_date + timedelta(days=7)
    elif any(_contains(text, phrase) for phrase in _TOMORROW):
        resolved_date = local_date + timedelta(days=1)
    elif any(_contains(text, phrase) for phrase in _TODAY):
        resolved_date = local_date
    else:
        resolved_date = None
        for weekday, number in _WEEKDAYS.items():
            suffix = r"(?:الياي|الياية|الجاي)" if weekday[0] != "m" else r"next"
            pattern = rf"(?:next\s+{weekday}|{weekday}\s+{suffix})"
            if re.search(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE):
                days = (number - local_date.weekday()) % 7 or 7
                resolved_date = local_date + timedelta(days=days)
                break

    relative_to = None
    offset = None
    relative = re.search(
        r"(?P<direction>قبل|بعد|عقب)\s+(?P<event>الاجتماع|الموعد)\s+"
        r"(?P<amount>بساعة|بنص ساعة|بربع ساعة|بساعتين)(?!\w)",
        text,
    )
    if relative:
        minutes = {"بساعة": 60, "بنص ساعة": 30, "بربع ساعة": 15, "بساعتين": 120}
        offset = minutes[relative.group("amount")]
        if relative.group("direction") == "قبل":
            offset = -offset
        relative_to = relative.group("event")

    return TemporalResolution(resolved_date, relative_to, offset)


# Original-script alternatives used only for exact, trailing entity cleanup.
TEMPORAL_SUFFIX_PATTERN = re.compile(
    r"\s+(?:عقب\s+(?:باچر|باجر|باكر)|بعد\s+(?:باچر|باجر|بكرة)|"
    r"باچر|باجر|باكر|بكرة|غدا|غداً|tomorrow)\s*$",
    re.IGNORECASE,
)


def remove_temporal_entity_suffix(value: str, source_text: str) -> str:
    """Remove an exact trailing phrase only when that phrase occurs in the action."""
    match = TEMPORAL_SUFFIX_PATTERN.search(value)
    if match is None or match.group(0).strip().casefold() not in source_text.casefold():
        return value
    cleaned = value[: match.start()].rstrip()
    return cleaned or value
