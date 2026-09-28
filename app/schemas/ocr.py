from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OCRPageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    page_number: int = Field(ge=1)
    text: str


class OCRResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    success: Literal[True] = True
    filename: str
    page_count: int = Field(ge=1)
    text: str
    pages: list[OCRPageResponse]
