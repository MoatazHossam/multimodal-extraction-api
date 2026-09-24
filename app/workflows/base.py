from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class Workflow(ABC):
    """A named extraction prompt and its independently validated output schema."""

    name: str
    output_model: type[BaseModel]

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Instructions sent to the AI provider for this workflow."""

    def output_json_schema(self) -> dict[str, Any]:
        return self.output_model.model_json_schema()

    def validate_output(self, data: dict[str, Any]) -> BaseModel:
        return self.output_model.model_validate(data)


class WorkflowNotFoundError(LookupError):
    """Raised when a requested workflow is not registered."""


class WorkflowRegistry:
    def __init__(self, workflows: list[Workflow]) -> None:
        self._workflows = {workflow.name: workflow for workflow in workflows}

    def get(self, name: str) -> Workflow:
        try:
            return self._workflows[name]
        except KeyError as exc:
            raise WorkflowNotFoundError(f"Unsupported workflow: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._workflows))

