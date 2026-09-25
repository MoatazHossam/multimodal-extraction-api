from app.schemas.responses import ActionDetectionResult
from app.workflows.base import Workflow


class ActionDetectionWorkflow(Workflow):
    """Detect ordered, actionable instructions without extracting their parameters."""

    name = "action_detection"
    output_model = ActionDetectionResult

    @property
    def system_prompt(self) -> str:
        return (
            "Detect every actionable instruction in the supplied Arabic or English text. "
            "Return only JSON conforming exactly to the supplied schema, with one actions item "
            "per distinct instruction and in the same order as the source. Classify only as: "
            "create_task for explicit work to assign or complete; create_meeting for scheduling "
            "a meeting or appointment involving participants; send_email for an explicit request "
            "to send or write email; create_request for registering a business, service, or "
            "assistance request or case; create_reminder for a request to remind the user; "
            "create_note for storing information as a note without another action; follow_up for "
            "an explicit follow-up regarding a person, proposal, request, or case; or unknown when "
            "there is no clear supported executable action. Preserve each relevant instruction in "
            "source_text in its original language. Do not invent information. Do not treat grammar "
            "or text correction as an action. If the whole input is non-actionable, return exactly "
            "one unknown action containing the input as source_text. Do not extract detailed "
            "parameters and do not combine separate actions."
        )
