import asyncio
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, UnidentifiedImageError

from app.ocr.provider import OCRProvider
from app.schemas.ocr import OCRPageResponse, OCRResponse

MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_PDF_PAGES = 20
SUPPORTED_TYPES = {
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".png": {"image/png"},
    ".webp": {"image/webp"},
    ".pdf": {"application/pdf"},
}


class OCRInputError(ValueError):
    pass


class OCRLimitError(OCRInputError):
    pass


class OCRService:
    def __init__(self, provider: OCRProvider) -> None:
        self._provider = provider
        # A single request/page enters the CPU-heavy engine at a time on the POC VPS.
        self._lock = asyncio.Lock()

    @staticmethod
    def validate_metadata(filename: str | None, content_type: str | None) -> str:
        safe_name = Path(filename or "").name
        suffix = Path(safe_name).suffix.lower()
        if not safe_name or suffix not in SUPPORTED_TYPES:
            raise OCRInputError("Unsupported file extension")
        if content_type not in SUPPORTED_TYPES[suffix]:
            raise OCRInputError("Unsupported MIME type")
        return safe_name

    async def extract(self, *, filename: str, content_type: str, data: bytes) -> OCRResponse:
        safe_name = self.validate_metadata(filename, content_type)
        if not data:
            raise OCRInputError("File is empty")
        if len(data) > MAX_FILE_SIZE:
            raise OCRLimitError("File exceeds the 15 MB limit")

        async with self._lock:
            if Path(safe_name).suffix.lower() == ".pdf":
                texts = await self._extract_pdf(data)
            else:
                texts = [await self._extract_image(data)]

        pages = [
            OCRPageResponse(page_number=index, text=text)
            for index, text in enumerate(texts, start=1)
        ]
        return OCRResponse(
            filename=safe_name,
            page_count=len(pages),
            text="\n".join(texts),
            pages=pages,
        )

    async def _extract_image(self, data: bytes) -> str:
        try:
            with Image.open(BytesIO(data)) as opened:
                opened.verify()
            with Image.open(BytesIO(data)) as opened:
                image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OCRInputError("Malformed image") from exc
        return await self._provider.recognize(image)

    async def _extract_pdf(self, data: bytes) -> list[str]:
        try:
            document = pdfium.PdfDocument(data)
            page_count = len(document)
        except Exception as exc:
            raise OCRInputError("Malformed PDF") from exc
        if page_count == 0:
            document.close()
            raise OCRInputError("Malformed PDF")
        if page_count > MAX_PDF_PAGES:
            document.close()
            raise OCRLimitError("PDF exceeds the 20 page limit")

        texts: list[str] = []
        try:
            for index in range(page_count):
                page = document[index]
                try:
                    try:
                        bitmap = page.render(scale=2)
                        image = bitmap.to_pil().convert("RGB")
                    except Exception as exc:
                        raise OCRInputError("Malformed PDF") from exc
                finally:
                    page.close()
                try:
                    texts.append(await self._provider.recognize(image))
                finally:
                    image.close()
                    bitmap.close()
        finally:
            document.close()
        return texts
