from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_WITH_WARNING = "APPROVE_WITH_WARNING"
    REWRITE = "REWRITE"
    BLOCK = "BLOCK"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardRequest(BaseModel):
    feature_id: str = Field(min_length=1, description="Selects schema and PII rules")
    user_prompt: str
    llm_output: str
    context: str | None = Field(
        default=None, description="Optional source material the output should be grounded in"
    )


class Finding(BaseModel):
    policy: str
    severity: Severity
    evidence: str
    action: Decision
    reason: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class GuardResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    decision: Decision
    risk_score: float = Field(ge=0, le=1)
    findings: list[Finding] = Field(default_factory=list)
    final_output: str | None = None
