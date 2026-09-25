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
            "source_text as the smallest complete clause from the input that expresses only that "
            "action. Do not include words or clauses belonging to another detected action. "
            "Preserve the original language and wording, and do not paraphrase unnecessarily or "
            "invent missing context. Pronouns and references may remain unresolved because "
            "parameter extraction will receive the original full text as context. Do not treat "
            "grammar or "
            "text correction as an action. If the whole input is non-actionable, return exactly "
            "one unknown action containing the input as source_text. Do not extract detailed "
            "parameters, "
            "combine separate actions, or change their order."
            " Treat Emirati/Gulf Arabic as a primary input style: سو لي or حط can introduce an "
            "action, طرش and contextually دز mean send, and ذكرني or نبهني request a reminder. "
            "An explicit اجتماع or meeting is create_meeting even when introduced by generic "
            "Emirati verbs such as سو، سوي، or حط. Examples: سو اجتماع باكر الساعة ١٠ means "
            "create_meeting; حط اجتماع ويا أحمد الأحد الياي means create_meeting; سو مهمة "
            "لأحمد يراجع التقرير means create_task. "
            "For example, سو لي اجتماع مع أحمد باچر الساعة عشر وطرش له إيميل بالتفاصيل "
            "وذكرني قبل الاجتماع بساعة contains create_meeting, send_email, and "
            "create_reminder actions."
        )
