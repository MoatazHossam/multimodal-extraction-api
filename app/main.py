import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.ai.factory import create_ai_provider
from app.api.v1.actions import router as actions_router
from app.api.v1.extraction import router as extraction_router
from app.api.v1.ocr import router as ocr_router
from app.api.v1.text import router as text_router
from app.config import Settings, get_settings
from app.ocr.paddle_provider import PaddleOCRProvider
from app.ocr.provider import OCRProvider
from app.schemas.responses import HealthResponse
from app.services.action_detection_service import ActionDetectionService
from app.services.action_parameter_extraction_service import ActionParameterExtractionService
from app.services.action_parsing_service import ActionParsingService
from app.services.extraction_service import ExtractionService
from app.services.ocr_service import OCRService
from app.services.text_correction_service import TextCorrectionService
from app.workflows.action_detection import ActionDetectionWorkflow
from app.workflows.assistance_request import AssistanceRequestWorkflow
from app.workflows.base import WorkflowRegistry
from app.workflows.email_parameters import EmailParameterWorkflow
from app.workflows.meeting_parameters import MeetingParameterWorkflow
from app.workflows.reminder_parameters import ReminderParameterWorkflow
from app.workflows.task_parameters import TaskParameterWorkflow
from app.workflows.text_correction import TextCorrectionWorkflow


def create_app(
    settings: Settings | None = None,
    *,
    ocr_provider: OCRProvider | None = None,
    ocr_provider_factory: type[OCRProvider] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    provider = create_ai_provider(settings)
    registry = WorkflowRegistry(
        [
            AssistanceRequestWorkflow(),
            ActionDetectionWorkflow(),
            TaskParameterWorkflow(),
            MeetingParameterWorkflow(),
            EmailParameterWorkflow(),
            ReminderParameterWorkflow(),
            TextCorrectionWorkflow("formal"),
            TextCorrectionWorkflow("asr_repair"),
            TextCorrectionWorkflow("asr_formal"),
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
        app.state.text_correction_service = TextCorrectionService(provider, registry)
        selected_ocr_provider = ocr_provider
        if selected_ocr_provider is None and ocr_provider_factory is not None:
            # Native model initialization happens once per application process,
            # during startup rather than on the first request or for every page.
            selected_ocr_provider = await asyncio.to_thread(ocr_provider_factory)
        if selected_ocr_provider is not None:
            app.state.ocr_service = OCRService(selected_ocr_provider)
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
    application.include_router(text_router)
    application.include_router(ocr_router)
    return application


app = create_app(ocr_provider_factory=PaddleOCRProvider)
