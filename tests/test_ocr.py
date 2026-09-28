import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ocr.paddle_provider import PaddleOCRProvider
from app.ocr.provider import OCRProvider, OCRProviderError
from app.services.ocr_service import OCRLimitError, OCRService


class StubOCRProvider(OCRProvider):
    def __init__(self, texts: list[str] | None = None, error: Exception | None = None) -> None:
        self.texts = iter(texts or ["recognized"])
        self.error = error

    async def recognize(self, image: Image.Image) -> str:
        if self.error:
            raise self.error
        return next(self.texts)


def image_bytes(format_name: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 8), "white").save(output, format=format_name)
    return output.getvalue()


def test_paddle_provider_uses_mobile_models_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    construction: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs: object) -> None:
            construction.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))

    PaddleOCRProvider()

    assert construction["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert construction["text_recognition_model_name"] == "arabic_PP-OCRv5_mobile_rec"
    assert construction["device"] == "cpu"
    assert "lang" not in construction
    assert "ocr_version" not in construction


@pytest.mark.parametrize(
    ("filename", "mime", "format_name"),
    [("scan.jpg", "image/jpeg", "JPEG"), ("scan.png", "image/png", "PNG")],
)
def test_image_uploads(filename: str, mime: str, format_name: str) -> None:
    app = create_app(ocr_provider=StubOCRProvider(["مرحبا ABC ١٢ 34!"]))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ocr/extract",
            files={"file": (filename, image_bytes(format_name), mime)},
        )
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "filename": filename,
        "page_count": 1,
        "text": "مرحبا ABC ١٢ 34!",
        "pages": [{"page_number": 1, "text": "مرحبا ABC ١٢ 34!"}],
    }


@pytest.mark.parametrize(
    ("filename", "mime"),
    [("scan.gif", "image/gif"), ("scan.png", "image/jpeg")],
)
def test_unsupported_extension_or_mime(filename: str, mime: str) -> None:
    with TestClient(create_app(ocr_provider=StubOCRProvider())) as client:
        response = client.post("/api/v1/ocr/extract", files={"file": (filename, b"data", mime)})
    assert response.status_code == 422


def test_empty_file() -> None:
    with TestClient(create_app(ocr_provider=StubOCRProvider())) as client:
        response = client.post(
            "/api/v1/ocr/extract", files={"file": ("scan.png", b"", "image/png")}
        )
    assert response.status_code == 422


def test_file_too_large() -> None:
    with TestClient(create_app(ocr_provider=StubOCRProvider())) as client:
        response = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("scan.jpg", b"x" * (15 * 1024 * 1024 + 1), "image/jpeg")},
        )
    assert response.status_code == 413


def test_provider_failure_is_generic_500() -> None:
    app = create_app(ocr_provider=StubOCRProvider(error=OCRProviderError("secret path")))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/ocr/extract",
            files={"file": ("scan.png", image_bytes("PNG"), "image/png")},
        )
    assert response.status_code == 500
    assert response.json() == {"detail": "OCR processing failed"}


class FakeBitmap:
    def __init__(self, number: int) -> None:
        self.number = number

    def to_pil(self) -> Image.Image:
        return Image.new("RGB", (1, 1), (self.number, 0, 0))

    def close(self) -> None:
        pass


class FakePage:
    def __init__(self, number: int) -> None:
        self.number = number

    def render(self, scale: int) -> FakeBitmap:
        assert scale == 2
        return FakeBitmap(self.number)

    def close(self) -> None:
        pass


class FakeDocument:
    def __init__(self, _: bytes, page_count: int = 3) -> None:
        self.page_count = page_count

    def __len__(self) -> int:
        return self.page_count

    def __getitem__(self, index: int) -> FakePage:
        return FakePage(index)

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_multi_page_pdf_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ocr_service.pdfium.PdfDocument", FakeDocument)
    result = await OCRService(StubOCRProvider(["first", "second", "third"])).extract(
        filename="doc.pdf", content_type="application/pdf", data=b"%PDF-fake"
    )
    assert [page.text for page in result.pages] == ["first", "second", "third"]
    assert result.text == "first\nsecond\nthird"


@pytest.mark.asyncio
async def test_pdf_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class LargeDocument(FakeDocument):
        def __init__(self, data: bytes) -> None:
            super().__init__(data, page_count=21)

    monkeypatch.setattr("app.services.ocr_service.pdfium.PdfDocument", LargeDocument)
    with pytest.raises(OCRLimitError, match="20 page"):
        await OCRService(StubOCRProvider()).extract(
            filename="doc.pdf", content_type="application/pdf", data=b"%PDF-fake"
        )
