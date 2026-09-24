from typing import Any, Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TextExtractionResponse(BaseModel):
    success: Literal[True] = True
    source_type: Literal["text"] = "text"
    text: str
    workflow: str
    data: dict[str, Any]
    missing_fields: list[str]

