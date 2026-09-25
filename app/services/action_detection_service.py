from pydantic import ValidationError

from app.ai.provider import AIProvider, AIProviderError
from app.schemas.requests import ActionDetectionRequest
from app.schemas.responses import ActionDetectionResponse
from app.services.source_text_normalizer import normalize_action_source_texts
from app.workflows.base import WorkflowRegistry


class ActionDetectionOutputError(AIProviderError):
    """Raised when provider output is not a valid action-detection result."""


class ActionDetectionService:
    def __init__(self, provider: AIProvider, workflows: WorkflowRegistry) -> None:
        self._provider = provider
        self._workflow = workflows.get("action_detection")

    async def detect(self, request: ActionDetectionRequest) -> ActionDetectionResponse:
        raw_data = await self._provider.extract_structured(
            system_prompt=self._workflow.system_prompt,
            user_text=request.text,
            output_schema=self._workflow.output_json_schema(),
        )
        try:
            output = self._workflow.validate_output(raw_data)
        except ValidationError as exc:
            raise ActionDetectionOutputError(
                "AI response did not match the action detection schema"
            ) from exc

        actions = normalize_action_source_texts(request.text, output.actions)
        return ActionDetectionResponse(text=request.text, actions=actions)
