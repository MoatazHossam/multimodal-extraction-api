from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.workflows.base import Workflow


class AssistanceRequestData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_number: str | None = Field(description="Case or file number, digits only")
    priority: Literal["urgent", "normal"] | None = Field(
        description="urgent only when the source explicitly indicates urgency"
    )
    request_type: Literal["financial_assistance", "other"] | None
    requester_name: str | None
    action_needed: str | None = Field(
        description="Concise requested action, preserving the source language"
    )


class AssistanceRequestWorkflow(Workflow):
    name = "assistance_request"
    output_model = AssistanceRequestData

    @property
    def system_prompt(self) -> str:
        return (
            "Extract an assistance request from the supplied Arabic or English text. "
            "Return only JSON conforming exactly to the supplied schema. Do not invent facts. "
            "Use null for information that is not present. Classify monetary, fee, tuition, rent, "
            "or bill support as financial_assistance; otherwise use other. Set priority to urgent "
            "only for explicit urgency. Preserve the source language in action_needed."
        )

