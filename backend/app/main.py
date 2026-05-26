"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI

from app.api import router as api_router
from app.config import get_settings
from app.logging_setup import configure_logging, get_logger


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info(
        "Starting %s app (env=%s, log_level=%s)",
        settings.app_name,
        settings.environment,
        settings.log_level,
    )
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(api_router)
    return app


app = create_app()
