from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

type ActionType = Literal[
    "create_task",
    "create_meeting",
    "send_email",
    "create_request",
    "create_reminder",
    "create_note",
    "follow_up",
    "unknown",
]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TextExtractionResponse(BaseModel):
    success: Literal[True] = True
    source_type: Literal["text"] = "text"
    text: str
    workflow: str
    data: dict[str, Any]
    missing_fields: list[str]


class DetectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    source_text: str = Field(min_length=1)

    @field_validator("source_text")
    @classmethod
    def reject_blank_source_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_text must not be blank")
        return value


class ActionDetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[DetectedAction] = Field(min_length=1)


class ActionDetectionResponse(ActionDetectionResult):
    success: Literal[True] = True
    text: str
