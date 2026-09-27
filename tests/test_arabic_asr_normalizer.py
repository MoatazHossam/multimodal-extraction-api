import pytest

from app.services.arabic_asr_normalizer import ArabicASRNormalizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("وحد اتنينتالتة", "واحد اثنين ثلاثة"),
        ("اتنين", "اثنين"),
        ("تنين", "اثنين"),
        ("تلاتة", "ثلاثة"),
        ("تلتة", "ثلاثة"),
        ("اربعة", "أربعة"),
        ("خمسه", "خمسة"),
        ("سته", "ستة"),
        ("سبعه", "سبعة"),
        ("تمانيه", "ثمانية"),
        ("تسعه", "تسعة"),
    ],
)
def test_normalizes_high_confidence_number_variants(text: str, expected: str) -> None:
    assert ArabicASRNormalizer().normalize(text) == expected


def test_leaves_unrelated_arabic_words_untouched() -> None:
    text = "سأحضر الاجتماع غدًا"

    assert ArabicASRNormalizer().normalize(text) == text


def test_only_splits_words_fully_composed_of_known_number_tokens() -> None:
    text = "اتنينمستندات"

    assert ArabicASRNormalizer().normalize(text) == text
