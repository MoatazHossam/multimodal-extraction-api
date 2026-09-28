from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.ocr.provider import OCRProviderError
from app.schemas.ocr import OCRResponse
from app.services.ocr_service import MAX_FILE_SIZE, OCRInputError, OCRLimitError, OCRService

router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


@router.post("/extract", response_model=OCRResponse)
async def extract_ocr(request: Request, file: Annotated[UploadFile, File()]) -> OCRResponse:
    service: OCRService = request.app.state.ocr_service
    try:
        filename = service.validate_metadata(file.filename, file.content_type)
        # Read one byte beyond the cap so oversized uploads are rejected without
        # buffering an unbounded request body in application memory.
        data = await file.read(MAX_FILE_SIZE + 1)
        return await service.extract(
            filename=filename,
            content_type=file.content_type or "",
            data=data,
        )
    except OCRLimitError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except OCRInputError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except OCRProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed",
        ) from exc
    finally:
        await file.close()
