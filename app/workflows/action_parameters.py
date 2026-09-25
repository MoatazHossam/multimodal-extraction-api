from abc import abstractmethod

from pydantic import BaseModel

from app.schemas.responses import (
    EmailParameters,
    MeetingParameters,
    ReminderParameters,
    TaskParameters,
)
from app.workflows.base import Workflow

COMMON_RULES = (
    "Return only JSON conforming exactly to the supplied schema. Extract only facts supported by "
    "the action clause or original full text. Never invent people, email addresses, dates, times, "
    "locations, priorities, or factual details. Use null or an empty list when information is not "
    "available. Preserve Arabic or English names and wording appropriately. Resolve pronouns from "
    "the original text only when the reference is reliable. Resolve relative date expressions "
    "using only the caller-provided reference_datetime and timezone; never use the server clock. "
    "Dates must be YYYY-MM-DD and times HH:MM in 24-hour format. If a value is ambiguous, return "
    "null rather than guessing."
)


class ActionParameterWorkflow(Workflow):
    @abstractmethod
    def missing_fields(self, parameters: BaseModel) -> list[str]:
        """Return execution-critical fields absent from validated parameters."""


class TaskParameterWorkflow(ActionParameterWorkflow):
    name = "create_task"
    output_model = TaskParameters

    @property
    def system_prompt(self) -> str:
        return (
            f"{COMMON_RULES} Extract task parameters. Do not assume a priority or assignee. "
            "A concise title and description may be derived only from the explicitly requested "
            "work."
        )

    def missing_fields(self, parameters: BaseModel) -> list[str]:
        task = TaskParameters.model_validate(parameters)
        return [] if task.title else ["title"]


class MeetingParameterWorkflow(ActionParameterWorkflow):
    name = "create_meeting"
    output_model = MeetingParameters

    @property
    def system_prompt(self) -> str:
        return (
            f"{COMMON_RULES} Extract meeting parameters. A concise title or agenda may be derived "
            "only from an explicitly stated meeting topic. Do not assume duration or location."
        )

    def missing_fields(self, parameters: BaseModel) -> list[str]:
        meeting = MeetingParameters.model_validate(parameters)
        missing = []
        if not meeting.attendees:
            missing.append("attendees")
        if meeting.date is None:
            missing.append("date")
        if meeting.time is None:
            missing.append("time")
        return missing


class EmailParameterWorkflow(ActionParameterWorkflow):
    name = "send_email"
    output_model = EmailParameters

    @property
    def system_prompt(self) -> str:
        return (
            f"{COMMON_RULES} Extract email parameters. Recipients may be names; never turn a name "
            "into a guessed email address. A concise subject and body draft may use only known "
            "context."
        )

    def missing_fields(self, parameters: BaseModel) -> list[str]:
        email = EmailParameters.model_validate(parameters)
        missing = []
        if not email.to:
            missing.append("to")
        if not email.body:
            missing.append("body")
        return missing


class ReminderParameterWorkflow(ActionParameterWorkflow):
    name = "create_reminder"
    output_model = ReminderParameters

    @property
    def system_prompt(self) -> str:
        return (
            f"{COMMON_RULES} Extract reminder parameters. For reminders relative to an event, keep "
            "date and time null unless safely known and set relative_to plus a signed "
            "offset_minutes (before is negative, after is positive). One hour before is -60."
        )

    def missing_fields(self, parameters: BaseModel) -> list[str]:
        reminder = ReminderParameters.model_validate(parameters)
        missing = []
        if not reminder.reminder_text:
            missing.append("reminder_text")
        has_absolute_schedule = reminder.date is not None and reminder.time is not None
        has_relative_schedule = (
            bool(reminder.relative_to) and reminder.offset_minutes is not None
        )
        if not has_absolute_schedule and not has_relative_schedule:
            missing.append("schedule")
        return missing
