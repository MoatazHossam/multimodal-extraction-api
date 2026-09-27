from fastapi import APIRouter, HTTPException, Request, status

from app.ai.provider import AIProviderError, AIProviderTimeoutError
from app.schemas.requests import TextCorrectionRequest
from app.schemas.responses import TextCorrectionResponse
from app.services.text_correction_service import TextCorrectionService

router = APIRouter(prefix="/api/v1/text", tags=["text"])


@router.post("/correct", response_model=TextCorrectionResponse)
async def correct_text(
    payload: TextCorrectionRequest, request: Request
) -> TextCorrectionResponse:
    service: TextCorrectionService = request.app.state.text_correction_service
    try:
        return await service.correct(payload)
    except AIProviderTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI text correction service failed",
        ) from exc
