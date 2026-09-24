from typing import Any

import pytest

from app.ai.provider import AIProvider
from app.schemas.requests import TextExtractionRequest
from app.services.extraction_service import ExtractionOutputError, ExtractionService
from app.workflows.assistance_request import AssistanceRequestWorkflow
from app.workflows.base import WorkflowNotFoundError, WorkflowRegistry


class StubProvider(AIProvider):
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.received_schema: dict[str, Any] | None = None

    async def extract_structured(
        self,
        *,
        system_prompt: str,
        user_text: str,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.received_schema = output_schema
        return self.result


@pytest.fixture
def registry() -> WorkflowRegistry:
    return WorkflowRegistry([AssistanceRequestWorkflow()])


async def test_extracts_and_reports_missing_fields(registry: WorkflowRegistry) -> None:
    provider = StubProvider(
        {
            "file_number": "5678",
            "priority": "urgent",
            "request_type": "financial_assistance",
            "requester_name": None,
            "action_needed": "مساعدة في دفع المصروفات الدراسية",
        }
    )
    service = ExtractionService(provider, registry)

    result = await service.extract_text(
        TextExtractionRequest(
            text="الملف رقم 5678 يحتاج مساعدة في دفع المصروفات الدراسية والأمر عاجل",
            workflow="assistance_request",
        )
    )

    assert result.data["file_number"] == "5678"
    assert result.missing_fields == ["requester_name"]
    assert provider.received_schema is not None


async def test_rejects_unknown_workflow(registry: WorkflowRegistry) -> None:
    service = ExtractionService(StubProvider({}), registry)
    with pytest.raises(WorkflowNotFoundError):
        await service.extract_text(TextExtractionRequest(text="hello", workflow="unknown"))


async def test_rejects_invalid_provider_output(registry: WorkflowRegistry) -> None:
    service = ExtractionService(StubProvider({"priority": "impossible"}), registry)
    with pytest.raises(ExtractionOutputError):
        await service.extract_text(
            TextExtractionRequest(text="hello", workflow="assistance_request")
        )

