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
    "Return only JSON conforming exactly to the supplied schema. Extract parameters only for "
    "action_source_text. prior_context exists only to resolve references, pronouns, people, "
    "events, or already-mentioned details. Never convert a different action in prior_context "
    "into parameters of the current action. Never use text after the current action. Never invent "
    "people, email addresses, dates, times, "
    "locations, priorities, or factual details. Use null or an empty list when information is not "
    "available. Preserve Arabic or English names and wording appropriately. Resolve pronouns from "
    "prior_context only when the reference is reliable. Resolve relative date expressions "
    "using only the caller-provided reference_datetime and timezone; never use the server clock. "
    "Dates must be YYYY-MM-DD and times HH:MM in 24-hour format. If a value is ambiguous, return "
    "null rather than guessing."
    " Emirati/Gulf Arabic is a first-class input style. باچر, باجر, باكر, بكرة, عقب باچر, "
    "اليوم, and الياي are scheduling terms, never parts of people's names. ويا means with."
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
            " Examples: كلف أحمد يراجع التقرير باچر; حط لسالم مهمة يخلص العرض اليوم; "
            "سو مهمة حق محمد يتابع الموضوع."
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
            "only from an explicit meeting topic in action_source_text, or from prior_context when "
            "action_source_text clearly references that topic. Email, reminder, and task "
            "instructions must never become a meeting title or agenda. Do not assume duration or "
            "location."
            " Examples: سو لي اجتماع مع أحمد باچر الساعة عشر; رتب اجتماع ويا محمد الأحد "
            "الياي; حط اجتماع بيني وبين سالم باجر."
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
            " Emirati send verbs include طرش and, when context is clear, دز. Examples: طرش "
            "إيميل لأحمد; طرش له إيميل بالتفاصيل; دز له إيميل."
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
            " Examples: ذكرني قبل الاجتماع بساعة; نبهني قبل الموعد بنص ساعة; عقب الاجتماع "
            "بساعة; بعد الاجتماع بساعتين. A quarter hour is 15, half an hour 30, one hour "
            "60, and two hours 120. Do not resolve قبلها without a reliable antecedent."
        )

    def missing_fields(self, parameters: BaseModel) -> list[str]:
        reminder = ReminderParameters.model_validate(parameters)
        missing = []
        if not reminder.reminder_text:
            missing.append("reminder_text")
        has_absolute_schedule = reminder.date is not None and reminder.time is not None
        has_relative_schedule = bool(reminder.relative_to) and reminder.offset_minutes is not None
        if not has_absolute_schedule and not has_relative_schedule:
            missing.append("schedule")
        return missing
