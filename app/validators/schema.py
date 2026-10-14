from pydantic import BaseModel, ValidationError

from app.models import Finding
from app.policies import Policy
from app.validators.feature_schemas import FEATURE_SCHEMAS

MAX_EVIDENCE_CHARS = 200


class SchemaValidator:
    """Checks structured LLM output against the schema registered for a feature.

    Severity, action, and fallback come from the `schema_mismatch` policy,
    so they can be changed in YAML.
    """

    def __init__(self, policy: Policy, schemas: dict[str, type[BaseModel]] | None = None):
        self.policy = policy
        self.schemas = FEATURE_SCHEMAS if schemas is None else schemas

    def validate(self, feature_id: str, llm_output: str) -> list[Finding]:
        schema = self.schemas.get(feature_id)
        if schema is None:
            return []  # free-text feature: nothing to validate

        try:
            schema.model_validate_json(llm_output)
        except ValidationError as exc:
            return [self._finding(llm_output, exc)]
        return []

    def _finding(self, llm_output: str, exc: ValidationError) -> Finding:
        reasons = []
        for error in exc.errors():
            if error["type"] == "json_invalid":
                reasons.append("output is not valid JSON")
                continue
            location = ".".join(str(part) for part in error["loc"]) or "(root)"
            reasons.append(f"{location}: {error['msg']}")

        return Finding(
            policy=self.policy.name,
            severity=self.policy.severity,
            # Always a real span of the output, so findings stay reviewable.
            evidence=llm_output.strip()[:MAX_EVIDENCE_CHARS],
            action=self.policy.action,
            reason="; ".join(reasons),
        )
