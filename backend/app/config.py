"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings sourced from environment variables and an optional `.env` file.

    Add new fields here as integrations land (LLM keys, Pinecone, E2B, Neon).
    Never commit a `.env` file — `.env.example` documents required keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "basketeer"
    environment: str = Field(default="dev", description="dev | staging | prod")
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
