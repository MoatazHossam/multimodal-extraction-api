import re
from functools import cache

_ARABIC_WORD = re.compile(r"[\u0621-\u064a]+")

# This intentionally small vocabulary contains only high-confidence number-word
# spellings. It is also the complete vocabulary used when splitting merged words.
_NUMBER_VARIANTS: dict[str, str] = {
    "واحد": "واحد",
    "وحد": "واحد",
    "اثنين": "اثنين",
    "اتنين": "اثنين",
    "تنين": "اثنين",
    "ثلاثة": "ثلاثة",
    "تلاتة": "ثلاثة",
    "تالتة": "ثلاثة",
    "تلتة": "ثلاثة",
    "أربعة": "أربعة",
    "اربعة": "أربعة",
    "خمسة": "خمسة",
    "خمسه": "خمسة",
    "ستة": "ستة",
    "سته": "ستة",
    "سبعة": "سبعة",
    "سبعه": "سبعة",
    "ثمانية": "ثمانية",
    "تمانية": "ثمانية",
    "تمانيه": "ثمانية",
    "تسعة": "تسعة",
    "تسعه": "تسعة",
}


class ArabicASRNormalizer:
    """Deterministically repair a conservative set of Arabic ASR number errors."""

    def normalize(self, text: str) -> str:
        return _ARABIC_WORD.sub(self._normalize_word, text)

    @staticmethod
    def _normalize_word(match: re.Match[str]) -> str:
        word = match.group()
        if replacement := _NUMBER_VARIANTS.get(word):
            return replacement

        segmentations = ArabicASRNormalizer._number_segmentations(word)
        if len(segmentations) != 1 or len(segmentations[0]) < 2:
            return word
        return " ".join(segmentations[0])

    @staticmethod
    def _number_segmentations(word: str) -> tuple[tuple[str, ...], ...]:
        """Return at most two segmentations, enough to detect ambiguity safely."""

        @cache
        def visit(offset: int) -> tuple[tuple[str, ...], ...]:
            if offset == len(word):
                return ((),)

            results: list[tuple[str, ...]] = []
            for variant, canonical in _NUMBER_VARIANTS.items():
                if not word.startswith(variant, offset):
                    continue
                for suffix in visit(offset + len(variant)):
                    candidate = (canonical, *suffix)
                    if candidate not in results:
                        results.append(candidate)
                    if len(results) == 2:
                        return tuple(results)
            return tuple(results)

        return visit(0)
