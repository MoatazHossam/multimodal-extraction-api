from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.ai.ollama_provider import OllamaProvider
from app.api.v1.actions import router as actions_router
from app.api.v1.extraction import router as extraction_router
from app.config import Settings, get_settings
from app.schemas.responses import HealthResponse
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import ActionParameterExtractionService
from app.services.action_parsing_service import ActionParsingService
from app.services.extraction_service import ExtractionService
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.assistance_request import AssistanceRequestWorkflow
from app.workflows.base import WorkflowRegistry
from app.workflows.email_parameters import EmailParameterWorkflow
from app.workflows.meeting_parameters import MeetingParameterWorkflow
from app.workflows.reminder_parameters import ReminderParameterWorkflow
from app.workflows.task_parameters import TaskParameterWorkflow


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    provider = OllamaProvider(
        base_url=str(settings.ollama_base_url),
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    registry = WorkflowRegistry(
        [
            AssistanceRequestWorkflow(),
            ActionDetectionWorkflow(),
            TaskParameterWorkflow(),
            MeetingParameterWorkflow(),
            EmailParameterWorkflow(),
            ReminderParameterWorkflow(),
        ]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.extraction_service = ExtractionService(provider, registry)
        app.state.action_detection_service = ActionDetectionService(provider, registry)
        app.state.action_parameter_service = ActionParameterExtractionService(provider, registry)
        app.state.action_parsing_service = ActionParsingService(
            app.state.action_detection_service, app.state.action_parameter_service
        )
        yield
        await provider.close()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse()

    application.include_router(extraction_router)
    application.include_router(actions_router)
    return application


app = create_app()
