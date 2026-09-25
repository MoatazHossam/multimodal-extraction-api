from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.responses import DetectedAction


class TextExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=50_000)
    workflow: str = Field(min_length=1, max_length=100)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ActionDetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=50_000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class ActionParameterExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_text: str = Field(min_length=1, max_length=50_000)
    action: DetectedAction
    reference_datetime: datetime
    timezone: str = Field(min_length=1, max_length=100)

    @field_validator("original_text")
    @classmethod
    def reject_blank_original_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("original_text must not be blank")
        return value

    @field_validator("reference_datetime")
    @classmethod
    def require_aware_reference_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reference_datetime must include a UTC offset")
        return value

    @field_validator("timezone")
    @classmethod
    def require_valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value
