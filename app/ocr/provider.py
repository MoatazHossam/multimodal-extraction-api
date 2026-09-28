from abc import ABC, abstractmethod

from PIL import Image


class OCRProviderError(RuntimeError):
    """Raised when an OCR engine cannot process a page."""


class OCRProvider(ABC):
    """Provider contract for recognizing one rasterized page at a time."""

    @abstractmethod
    async def recognize(self, image: Image.Image) -> str:
        """Return recognized text in reading order."""
