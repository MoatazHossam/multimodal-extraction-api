from fastapi import APIRouter, HTTPException, Request, status

from app.ai.provider import AIProviderError, AIProviderTimeoutError
from app.schemas.requests import TextExtractionRequest
from app.schemas.responses import TextExtractionResponse
from app.services.extraction_service import ExtractionService
from app.workflows.base import WorkflowNotFoundError

router = APIRouter(prefix="/api/v1/extract", tags=["extraction"])


@router.post("/text", response_model=TextExtractionResponse)
async def extract_text(
    payload: TextExtractionRequest, request: Request
) -> TextExtractionResponse:
    service: ExtractionService = request.app.state.extraction_service
    try:
        return await service.extract_text(payload)
    except WorkflowNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except AIProviderTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except AIProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI extraction service failed",
        ) from exc
