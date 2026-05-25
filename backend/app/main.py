"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI

from app.api import router as api_router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.include_router(api_router)
    return app


app = create_app()
