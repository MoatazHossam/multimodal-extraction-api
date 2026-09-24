from functools import lru_cache

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Multimodal Extraction API"
    app_version: str = "0.1.0"
    ollama_base_url: AnyHttpUrl = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="qwen3:1.7b", min_length=1)
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()

