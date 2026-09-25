from fastapi import APIRouter, HTTPException, Request, status

from app.ai.provider import AIProviderError, AIProviderTimeoutError
from app.schemas.requests import ActionDetectionRequest
from app.schemas.responses import ActionDetectionResponse
from app.services.action_detection_service import ActionDetectionService

router = APIRouter(prefix="/api/v1/actions", tags=["actions"])


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
