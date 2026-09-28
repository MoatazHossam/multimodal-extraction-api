import asyncio
from typing import Any

import numpy as np
from PIL import Image

from app.ocr.provider import OCRProvider, OCRProviderError


class PaddleOCRProvider(OCRProvider):
    """CPU-only PP-OCRv5 adapter. Paddle-specific values stay behind this boundary."""

    def __init__(self) -> None:
        # Importing here keeps module imports and mocked tests independent of Paddle's
        # sizeable native runtime. The production app constructs this once at startup.
        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(
            lang="ar",
            ocr_version="PP-OCRv5",
            device="cpu",
            enable_mkldnn=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    async def recognize(self, image: Image.Image) -> str:
        try:
            return await asyncio.to_thread(self._recognize_sync, image)
        except OCRProviderError:
            raise
        except Exception as exc:
            raise OCRProviderError("OCR engine failed") from exc

    def _recognize_sync(self, image: Image.Image) -> str:
        source = np.asarray(image.convert("RGB"))
        if hasattr(self._engine, "predict"):
            results = self._engine.predict(source)
            lines: list[str] = []
            for result in results:
                payload: Any = getattr(result, "json", result)
                if callable(payload):
                    payload = payload()
                if isinstance(payload, dict):
                    payload = payload.get("res", payload)
                    lines.extend(str(text) for text in payload.get("rec_texts", []))
            return "\n".join(lines)

        # Compatibility with PaddleOCR 2.x while deployments transition to 3.x.
        results = self._engine.ocr(source, cls=True)
        page = results[0] if results else []
        return "\n".join(str(line[1][0]) for line in page if line and line[1])
