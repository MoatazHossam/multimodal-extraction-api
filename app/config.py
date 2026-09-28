from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Multimodal Extraction API"
    app_version: str = "0.1.0"
    ai_provider: Literal["ollama", "groq"] = "ollama"
    ollama_base_url: AnyHttpUrl = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="qwen3:1.7b", min_length=1)
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    groq_api_key: SecretStr | None = None
    groq_model: str = Field(default="qwen/qwen3.8-27b", min_length=1)
    groq_base_url: AnyHttpUrl = Field(default="https://api.groq.com/openai/v1")
    groq_timeout_seconds: float = Field(default=60.0, gt=0)
    paddle_text_detection_model: str = Field(default="PP-OCRv5_mobile_det", min_length=1)
    paddle_text_recognition_model: str = Field(
        default="arabic_PP-OCRv5_mobile_rec", min_length=1
    )

    @model_validator(mode="after")
    def require_selected_provider_credentials(self) -> "Settings":
        if self.ai_provider == "groq" and (
            self.groq_api_key is None or not self.groq_api_key.get_secret_value().strip()
        ):
            raise ValueError("GROQ_API_KEY is required when AI_PROVIDER=groq")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
