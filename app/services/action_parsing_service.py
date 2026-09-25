from app.schemas.requests import (
    ActionDetectionRequest,
    ActionParameterExtractionRequest,
    ActionParsingRequest,
)
from app.schemas.responses import (
    ActionParsingResponse,
    ParsedSupportedAction,
    ParsedUnsupportedAction,
)
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import ActionParameterExtractionService

SUPPORTED_PARAMETER_ACTIONS = {
    "create_task",
    "create_meeting",
    "send_email",
    "create_reminder",
}


class ActionParsingService:
    """Detect actions and sequentially extract parameters for supported actions."""

    def __init__(
        self,
        detection_service: ActionDetectionService,
        parameter_service: ActionParameterExtractionService,
    ) -> None:
        self._detection_service = detection_service
        self._parameter_service = parameter_service

    async def parse(self, request: ActionParsingRequest) -> ActionParsingResponse:
        detection = await self._detection_service.detect(ActionDetectionRequest(text=request.text))
        parsed_actions: list[ParsedSupportedAction | ParsedUnsupportedAction] = []

        # Deliberately await each extraction before starting the next one. The local model is
        # resource constrained, and action order is also significant for contextual references.
        for action in detection.actions:
            if action.action_type not in SUPPORTED_PARAMETER_ACTIONS:
                parsed_actions.append(
                    ParsedUnsupportedAction(
                        action_type=action.action_type,
                        source_text=action.source_text,
                    )
                )
                continue

            extracted = await self._parameter_service.extract(
                ActionParameterExtractionRequest(
                    original_text=request.text,
                    action=action,
                    reference_datetime=request.reference_datetime,
                    timezone=request.timezone,
                )
            )
            parsed_actions.append(
                ParsedSupportedAction(
                    action_type=extracted.action_type,
                    source_text=extracted.source_text,
                    parameters=extracted.parameters,
                    missing_fields=extracted.missing_fields,
                )
            )

        return ActionParsingResponse(text=request.text, actions=parsed_actions)
