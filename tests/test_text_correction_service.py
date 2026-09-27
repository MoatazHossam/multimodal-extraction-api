from typing import Any

import pytest

from app.ai.provider import AIProvider
from app.schemas.requests import TextCorrectionRequest
from app.services.text_correction_service import (
    TextCorrectionOutputError,
    TextCorrectionService,
)
from app.workflows.base import WorkflowRegistry
from app.workflows.text_correction import TextCorrectionWorkflow


class CorrectionStubProvider(AIProvider):
    def __init__(self, corrected_text: str) -> None:
        self.corrected_text = corrected_text
        self.prompts: list[str] = []
        self.user_texts: list[str] = []

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.prompts.append(system_prompt)
        self.user_texts.append(user_text)
        return {"corrected_text": self.corrected_text}


def correction_registry() -> WorkflowRegistry:
    return WorkflowRegistry(
        [
            TextCorrectionWorkflow("formal"),
            TextCorrectionWorkflow("asr_repair"),
            TextCorrectionWorkflow("asr_formal"),
        ]
    )


@pytest.mark.parametrize("mode", ["asr_repair", "asr_formal"])
async def test_repairs_merged_arabic_number_words(mode: str) -> None:
    provider = CorrectionStubProvider("واحد اثنين ثلاثة")
    service = TextCorrectionService(provider, correction_registry())

    result = await service.correct(TextCorrectionRequest(text="وحد اتنينتالتة", mode=mode))

    assert result.corrected_text == "واحد اثنين ثلاثة"
    assert result.original_text == "وحد اتنينتالتة"
    assert result.changed is True
    assert result.mode == mode
    assert provider.user_texts == ["واحد اثنين ثلاثة"]


async def test_formal_mode_does_not_apply_asr_normalization() -> None:
    text = "وحد اتنينتالتة"
    provider = CorrectionStubProvider(text)
    service = TextCorrectionService(provider, correction_registry())

    result = await service.correct(TextCorrectionRequest(text=text, mode="formal"))

    assert provider.user_texts == [text]
    assert result.original_text == text
    assert result.changed is False


async def test_formalizes_dialectal_arabic_professionally() -> None:
    original = "انا رايح الاجتماع بكره و هبعت لاحمد التفاصيل"
    corrected = "سأذهب إلى الاجتماع غدًا، وسأرسل التفاصيل إلى أحمد."
    provider = CorrectionStubProvider(corrected)
    service = TextCorrectionService(provider, correction_registry())

    result = await service.correct(TextCorrectionRequest(text=original, mode="formal"))

    assert result.corrected_text == corrected
    assert result.original_text == original
    assert "professional Modern Standard Arabic" in provider.prompts[0]


async def test_already_correct_arabic_remains_unchanged() -> None:
    text = "سأحضر الاجتماع غدًا."
    service = TextCorrectionService(CorrectionStubProvider(text), correction_registry())

    result = await service.correct(TextCorrectionRequest(text=text, mode="formal"))

    assert result.corrected_text == text
    assert result.changed is False


async def test_name_and_existing_id_are_preserved() -> None:
    original = "الأستاذة مريم ستراجع الملف AB-2048 غدًا."
    service = TextCorrectionService(CorrectionStubProvider(original), correction_registry())

    result = await service.correct(TextCorrectionRequest(text=original, mode="formal"))

    assert "مريم" in result.corrected_text
    assert "AB-2048" in result.corrected_text


async def test_rejects_invented_or_changed_numeric_values() -> None:
    provider = CorrectionStubProvider("راجع الملف 2049 الساعة 10")
    service = TextCorrectionService(provider, correction_registry())

    with pytest.raises(TextCorrectionOutputError):
        await service.correct(TextCorrectionRequest(text="راجع الملف 2048", mode="asr_repair"))


async def test_changed_is_computed_in_application_not_accepted_from_provider() -> None:
    class ProviderWithFakeChanged(CorrectionStubProvider):
        async def extract_structured(
            self,
            *,
            system_prompt: str,
            user_text: str,
            output_schema: dict[str, Any],
        ) -> dict[str, Any]:
            return {"corrected_text": user_text, "changed": True}

    service = TextCorrectionService(ProviderWithFakeChanged("unused"), correction_registry())

    with pytest.raises(TextCorrectionOutputError):
        await service.correct(TextCorrectionRequest(text="نص صحيح.", mode="formal"))
