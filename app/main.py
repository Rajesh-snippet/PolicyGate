from fastapi import FastAPI

from app.config import get_settings
from app.models import Decision, GuardRequest, GuardResponse

settings = get_settings()
app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}


@app.post("/guard", response_model=GuardResponse)
def guard(request: GuardRequest) -> GuardResponse:
    # Day 1 stub: returns a fixed decision.
    # Later days replace this with the real pipeline:
    # schema -> PII -> rules -> judge -> decision engine.
    return GuardResponse(
        decision=Decision.HUMAN_REVIEW,
        risk_score=0.8,
        findings=[],
        final_output=None,
    )
