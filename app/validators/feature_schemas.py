"""Pydantic schemas for AI features that must return structured JSON.

Add a model here and register it under its feature_id. Features with no
entry are free text and skip schema validation.
"""
from pydantic import BaseModel, ConfigDict, Field


class AnswerOutput(BaseModel):
    # strict: no type coercion ("0.72" is not a float); forbid: no extra fields.
    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


FEATURE_SCHEMAS: dict[str, type[BaseModel]] = {
    "structured_answer": AnswerOutput,
}
