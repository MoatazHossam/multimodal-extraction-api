from pydantic import ValidationError

from app.ai.provider import AIProvider, AIProviderError
from app.schemas.requests import TextExtractionRequest
from app.schemas.responses import TextExtractionResponse
from app.workflows.base import WorkflowRegistry


class ExtractionOutputError(AIProviderError):
    """Raised when provider output does not satisfy the selected workflow."""


class ExtractionService:
    def __init__(self, provider: AIProvider, workflows: WorkflowRegistry) -> None:
        self._provider = provider
        self._workflows = workflows

    async def extract_text(self, request: TextExtractionRequest) -> TextExtractionResponse:
        workflow = self._workflows.get(request.workflow)
        raw_data = await self._provider.extract_structured(
            system_prompt=workflow.system_prompt,
            user_text=request.text,
            output_schema=workflow.output_json_schema(),
        )
        try:
            output = workflow.validate_output(raw_data)
        except ValidationError as exc:
            raise ExtractionOutputError(
                "AI response did not match the workflow schema"
            ) from exc

        data = output.model_dump()
        missing_fields = [name for name, value in data.items() if value is None]
        return TextExtractionResponse(
            text=request.text,
            workflow=workflow.name,
            data=data,
            missing_fields=missing_fields,
        )

