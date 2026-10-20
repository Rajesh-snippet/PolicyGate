import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.judge.client import JudgeClient
from app.judge.prompts import build_prompt
from app.models import Finding, Severity
from app.pipeline import DeterministicResult
from app.policies import Policy, PolicyRegistry, Strategy

# Features whose outputs get two judge passes and always reach the judge.
HIGH_RISK_FEATURES: set[str] = {"health_assistant"}

SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


class JudgeError(Exception):
    """A judge call failed or returned something unusable."""


class JudgeViolation(BaseModel):
    policy: str
    severity: Severity
    evidence: str
    reason: str = ""
    confidence: float = Field(ge=0, le=1)

    @field_validator("severity", mode="before")
    @classmethod
    def _lower(cls, value):
        return value.lower() if isinstance(value, str) else value


class JudgeVerdict(BaseModel):
    violations: list[JudgeViolation] = Field(default_factory=list)


@dataclass
class JudgeResult:
    findings: list[Finding] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None
    judged_policies: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)  # policies where passes split
    failed_policies: dict[str, str] = field(default_factory=dict)  # policy -> error
    unverified_discarded: int = 0  # findings dropped: evidence not in the output
    latency_ms: float = 0.0

    @property
    def failed(self) -> bool:
        return bool(self.failed_policies)

    @property
    def disagreed(self) -> bool:
        return bool(self.disagreements)


@dataclass
class _PolicyOutcome:
    findings: list[Finding] = field(default_factory=list)
    disagreed: bool = False
    discarded: int = 0
    error: str | None = None


def parse_verdict(raw: str) -> JudgeVerdict:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])  # JSON wrapped in prose

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return JudgeVerdict.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = exc
    raise JudgeError(f"judge returned invalid JSON: {last_error}")


class LLMJudge:
    """Reviews output against llm_judge and hybrid policies, one prompt per policy.

    - Every finding must quote evidence that really appears in the output;
      anything else is discarded.
    - The judge can lower a finding's severity but never raise it above the
      policy's own severity.
    - High-risk features get two independent passes; a split goes to review.
    - Judge failures are reported, never swallowed, so the decision engine can
      fail closed.
    """

    def __init__(
        self,
        policies: list[Policy],
        client: JudgeClient,
        second_client: JudgeClient | None = None,
        high_risk_features: set[str] | None = None,
        max_workers: int = 4,
    ):
        self.policies = [
            p for p in policies if p.strategy in (Strategy.LLM_JUDGE, Strategy.HYBRID)
        ]
        self.client = client
        # Ideally a different model; the same client only catches sampling noise.
        self.second_client = second_client or client
        self.high_risk_features = HIGH_RISK_FEATURES if high_risk_features is None else high_risk_features
        self.max_workers = max_workers

    @classmethod
    def from_registry(cls, registry: PolicyRegistry, client: JudgeClient, **kwargs) -> "LLMJudge":
        return cls(registry.all(), client, **kwargs)

    def is_high_risk(self, feature_id: str) -> bool:
        return feature_id in self.high_risk_features

    def should_run(
        self, feature_id: str, deterministic: DeterministicResult | None
    ) -> tuple[bool, str | None]:
        if deterministic is None:
            return True, None
        if deterministic.hard_block:
            return False, "hard block from deterministic checks"
        if deterministic.findings and not self.is_high_risk(feature_id):
            return False, "deterministic findings already decide a low-risk feature"
        return True, None

    def review(
        self,
        feature_id: str,
        user_prompt: str,
        llm_output: str,
        context: str | None = None,
        deterministic: DeterministicResult | None = None,
    ) -> JudgeResult:
        started = time.perf_counter()
        run, reason = self.should_run(feature_id, deterministic)
        if not run:
            return JudgeResult(skipped=True, skip_reason=reason)

        flagged = {f.policy for f in deterministic.findings} if deterministic else set()
        policies = [p for p in self.policies if p.name not in flagged]
        if not policies:
            return JudgeResult(skipped=True, skip_reason="no policies left for the judge")

        high_risk = self.is_high_risk(feature_id)

        def judge_one(policy: Policy) -> _PolicyOutcome:
            return self._judge_policy(policy, user_prompt, llm_output, context, high_risk)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            outcomes = list(pool.map(judge_one, policies))  # keeps policy order

        result = JudgeResult(judged_policies=[p.name for p in policies])
        for policy, outcome in zip(policies, outcomes):
            result.unverified_discarded += outcome.discarded
            if outcome.error:
                result.failed_policies[policy.name] = outcome.error
                continue
            result.findings.extend(outcome.findings)
            if outcome.disagreed:
                result.disagreements.append(policy.name)
        result.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return result

    def _judge_policy(
        self, policy: Policy, user_prompt: str, llm_output: str, context: str | None, high_risk: bool
    ) -> _PolicyOutcome:
        clients = [self.client, self.second_client] if high_risk else [self.client]
        passes: list[list[Finding]] = []
        discarded = 0
        try:
            for client in clients:
                findings, dropped = self._single_pass(
                    client, policy, user_prompt, llm_output, context
                )
                passes.append(findings)
                discarded += dropped
        except JudgeError as exc:
            return _PolicyOutcome(error=str(exc), discarded=discarded)

        disagreed = len(passes) == 2 and (bool(passes[0]) != bool(passes[1]))
        return _PolicyOutcome(
            findings=self._merge(passes), disagreed=disagreed, discarded=discarded
        )

    def _single_pass(
        self,
        client: JudgeClient,
        policy: Policy,
        user_prompt: str,
        llm_output: str,
        context: str | None,
    ) -> tuple[list[Finding], int]:
        system, user = build_prompt(policy, user_prompt, llm_output, context)
        try:
            raw = client.complete(system, user)
        except Exception as exc:  # timeouts, rate limits, network errors
            raise JudgeError(f"judge call failed: {exc}") from exc

        verdict = parse_verdict(raw)
        findings: list[Finding] = []
        dropped = 0
        for violation in verdict.violations:
            evidence_ok = bool(violation.evidence.strip()) and violation.evidence in llm_output
            if violation.policy != policy.name or not evidence_ok:
                dropped += 1
                continue
            severity = min(
                violation.severity, policy.severity, key=lambda s: SEVERITY_ORDER.index(s)
            )
            findings.append(
                Finding(
                    policy=policy.name,
                    severity=severity,
                    evidence=violation.evidence,
                    action=policy.action,
                    reason=violation.reason or None,
                    confidence=violation.confidence,
                )
            )
        return findings, dropped

    @staticmethod
    def _merge(passes: list[list[Finding]]) -> list[Finding]:
        merged: dict[str, Finding] = {}
        for findings in passes:
            for finding in findings:
                existing = merged.get(finding.evidence)
                # same quote from both passes: keep the more cautious confidence
                if existing is None or (finding.confidence or 0) < (existing.confidence or 0):
                    merged[finding.evidence] = finding
        return list(merged.values())
