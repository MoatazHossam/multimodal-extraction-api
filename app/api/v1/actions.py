from fastapi import APIRouter, HTTPException, Request, status

from app.ai.provider import AIProviderError, AIProviderTimeoutError
from app.schemas.requests import (
    ActionDetectionRequest,
    ActionParameterExtractionRequest,
    ActionParsingRequest,
)
from app.schemas.responses import (
    ActionDetectionResponse,
    ActionParameterExtractionResponse,
    ActionParsingResponse,
)
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import (
    ActionParameterExtractionService,
    UnsupportedActionTypeError,
)
from app.services.action_parsing_service import ActionParsingService

router = APIRouter(prefix="/api/v1/actions", tags=["actions"])


@router.post("/parse", response_model=ActionParsingResponse)
async def parse_actions(
    payload: ActionParsingRequest, request: Request
) -> ActionParsingResponse:
    service: ActionParsingService = request.app.state.action_parsing_service
    try:
        return await service.parse(payload)
    except AIProviderTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI action parsing service failed",
        ) from exc


@router.post("/detect", response_model=ActionDetectionResponse)
async def detect_actions(
    payload: ActionDetectionRequest, request: Request
) -> ActionDetectionResponse:
    service: ActionDetectionService = request.app.state.action_detection_service
    try:
        return await service.detect(payload)
    except AIProviderTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI action detection service failed",
        ) from exc


@router.post("/extract-parameters", response_model=ActionParameterExtractionResponse)
async def extract_action_parameters(
    payload: ActionParameterExtractionRequest, request: Request
) -> ActionParameterExtractionResponse:
    service: ActionParameterExtractionService = request.app.state.action_parameter_service
    try:
        return await service.extract(payload)
    except UnsupportedActionTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except AIProviderTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI action parameter extraction service failed",
        ) from exc
