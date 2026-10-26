from functools import lru_cache

from fastapi import Depends, FastAPI

from app.config import get_settings
from app.models import GuardRequest, GuardResponse
from app.service import GuardService, build_service

settings = get_settings()
app = FastAPI(title=settings.app_name)


@lru_cache
def get_service() -> GuardService:
    return build_service(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/guard", response_model=GuardResponse)
def guard(request: GuardRequest, service: GuardService = Depends(get_service)) -> GuardResponse:
    return service.guard(request)
