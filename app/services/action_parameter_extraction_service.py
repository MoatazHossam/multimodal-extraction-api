import json

from pydantic import ValidationError

from app.ai.provider import AIProvider, AIProviderError
from app.schemas.requests import ActionParameterExtractionRequest
from app.schemas.responses import ActionParameterExtractionResponse
from app.workflows.action_parameters import ActionParameterWorkflow
from app.workflows.base import WorkflowNotFoundError, WorkflowRegistry


class UnsupportedActionTypeError(ValueError):
    """Raised when parameter extraction is unavailable for an action type."""


class ActionParameterOutputError(AIProviderError):
    """Raised when provider output does not match the action-specific schema."""


class ActionParameterExtractionService:
    def __init__(self, provider: AIProvider, workflows: WorkflowRegistry) -> None:
        self._provider = provider
        self._workflows = workflows

    async def extract(
        self, request: ActionParameterExtractionRequest
    ) -> ActionParameterExtractionResponse:
        try:
            workflow = self._workflows.get(request.action.action_type)
        except WorkflowNotFoundError as exc:
            raise UnsupportedActionTypeError(
                f"Parameter extraction is not supported for {request.action.action_type}"
            ) from exc
        if not isinstance(workflow, ActionParameterWorkflow):
            raise UnsupportedActionTypeError(
                f"Parameter extraction is not supported for {request.action.action_type}"
            )

        context = json.dumps(
            {
                "original_text": request.original_text,
                "action_source_text": request.action.source_text,
                "action_type": request.action.action_type,
                "reference_datetime": request.reference_datetime.isoformat(),
                "timezone": request.timezone,
            },
            ensure_ascii=False,
        )
        raw_data = await self._provider.extract_structured(
            system_prompt=workflow.system_prompt,
            user_text=context,
            output_schema=workflow.output_json_schema(),
        )
        try:
            parameters = workflow.validate_output(raw_data)
        except ValidationError as exc:
            raise ActionParameterOutputError(
                "AI response did not match the action parameter schema"
            ) from exc

        return ActionParameterExtractionResponse(
            action_type=request.action.action_type,
            source_text=request.action.source_text,
            parameters=parameters,
            missing_fields=workflow.missing_fields(parameters),
        )
