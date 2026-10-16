"""Cheap deterministic checks, run before any LLM call."""
import time
from dataclasses import dataclass, field

from app.models import Decision, Finding
from app.policies import PolicyRegistry
from app.validators.pii import PIIDetector
from app.validators.rules import RuleEngine
from app.validators.schema import SchemaValidator


@dataclass
class DeterministicResult:
    findings: list[Finding]
    stage_latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def hard_block(self) -> bool:
        """True if any finding's policy action is BLOCK."""
        return any(f.action == Decision.BLOCK for f in self.findings)

    @property
    def skip_judge(self) -> bool:
        """A hard block is final, so the LLM judge is not worth calling."""
        return self.hard_block


class DeterministicChecks:
    """Runs schema, PII, and rule checks (cheap to expensive) and collects findings.

    All three stages always run so the audit trail has the full picture;
    they cost microseconds. What a hard block saves is the LLM judge call.
    """

    def __init__(
        self,
        schema_validator: SchemaValidator,
        pii_detector: PIIDetector,
        rule_engine: RuleEngine,
    ):
        self.stages = [
            ("schema", schema_validator.validate),
            ("pii", pii_detector.detect),
            ("rules", rule_engine.check),
        ]

    @classmethod
    def from_registry(cls, registry: PolicyRegistry) -> "DeterministicChecks":
        return cls(
            SchemaValidator(registry.get("schema_mismatch")),
            PIIDetector(registry.get("pii_leakage")),
            RuleEngine.from_registry(registry),
        )

    def run(self, feature_id: str, llm_output: str) -> DeterministicResult:
        findings: list[Finding] = []
        latency: dict[str, float] = {}
        for name, stage in self.stages:
            started = time.perf_counter()
            findings.extend(stage(feature_id, llm_output))
            latency[name] = round((time.perf_counter() - started) * 1000, 3)
        return DeterministicResult(findings=findings, stage_latency_ms=latency)