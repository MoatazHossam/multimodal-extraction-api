from datetime import date, time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

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


class TaskParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None
    assignees: list[str]
    due_date: date | None
    due_time: time | None
    priority: Literal["low", "normal", "high", "urgent"] | None
    description: str | None

    @field_serializer("due_time", when_used="json")
    def serialize_due_time(self, value: time | None) -> str | None:
        return value.strftime("%H:%M") if value is not None else None


class MeetingParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None
    attendees: list[str]
    date: date | None
    time: time | None
    duration_minutes: int | None = Field(ge=1)
    location: str | None
    agenda: str | None

    @field_serializer("time", when_used="json")
    def serialize_time(self, value: time | None) -> str | None:
        return value.strftime("%H:%M") if value is not None else None


class EmailParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    to: list[str]
    cc: list[str]
    subject: str | None
    body: str | None


class ReminderParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reminder_text: str | None
    date: date | None
    time: time | None
    relative_to: str | None
    offset_minutes: int | None

    @field_serializer("time", when_used="json")
    def serialize_time(self, value: time | None) -> str | None:
        return value.strftime("%H:%M") if value is not None else None


type ActionParameters = TaskParameters | MeetingParameters | EmailParameters | ReminderParameters


class ActionParameterExtractionResponse(BaseModel):
    success: Literal[True] = True
    action_type: Literal["create_task", "create_meeting", "send_email", "create_reminder"]
    source_text: str
    parameters: ActionParameters
    missing_fields: list[str]
