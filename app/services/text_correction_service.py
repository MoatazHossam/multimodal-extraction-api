import re
from collections import Counter

from pydantic import ValidationError

from app.ai.provider import AIProvider, AIProviderError
from app.schemas.requests import TextCorrectionRequest
from app.schemas.responses import TextCorrectionResponse
from app.services.arabic_asr_normalizer import ArabicASRNormalizer
from app.workflows.base import WorkflowRegistry


class TextCorrectionOutputError(AIProviderError):
    """Raised when corrected text is invalid or violates a deterministic fidelity rule."""


class TextCorrectionService:
    def __init__(self, provider: AIProvider, workflows: WorkflowRegistry) -> None:
        self._provider = provider
        self._workflows = workflows
        self._asr_normalizer = ArabicASRNormalizer()

    async def correct(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        workflow = self._workflows.get(f"text_correction_{request.mode}")
        llm_input = request.text
        if request.mode in ("asr_repair", "asr_formal"):
            llm_input = self._asr_normalizer.normalize(request.text)
        raw_data = await self._provider.extract_structured(
            system_prompt=workflow.system_prompt,
            user_text=llm_input,
            output_schema=workflow.output_json_schema(),
        )
        try:
            output = workflow.validate_output(raw_data)
        except ValidationError as exc:
            raise TextCorrectionOutputError(
                "AI response did not match the text correction schema"
            ) from exc

        corrected_text = output.corrected_text
        self._verify_literal_values(request.text, corrected_text)
        return TextCorrectionResponse(
            original_text=request.text,
            corrected_text=corrected_text,
            changed=corrected_text != request.text,
            mode=request.mode,
        )

    @staticmethod
    def _verify_literal_values(original: str, corrected: str) -> None:
        """Reject additions, removals, or substitutions of machine-detectable sensitive values."""
        value_patterns = (
            r"\d+",
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        )
        for pattern in value_patterns:
            if Counter(re.findall(pattern, original)) != Counter(re.findall(pattern, corrected)):
                raise TextCorrectionOutputError(
                    "AI response changed a protected number or email address"
                )
