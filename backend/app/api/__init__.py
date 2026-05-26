from fastapi import APIRouter

from app.api import ask, health

router = APIRouter()
router.include_router(health.router)
router.include_router(ask.router)
