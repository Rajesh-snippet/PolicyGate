"""The guardrail service: deterministic checks -> judge -> decision -> audit log."""
import logging
import time
from uuid import uuid4

from app.audit import AuditEntry, JsonlAuditLog, scrub, sha256_hex
from app.config import Settings
from app.decision import DecisionEngine
from app.judge.client import JudgeConfigError, UnavailableJudgeClient, build_judge_client
from app.judge.judge import LLMJudge
from app.models import Decision, Finding, GuardRequest, GuardResponse
from app.pipeline import DeterministicChecks
from app.policies import load_policies

logger = logging.getLogger("policygate")

# Until the rewrite and block flows exist (Day 8), only approved output is returned.
RELEASABLE = {Decision.APPROVE, Decision.APPROVE_WITH_WARNING}


def _scrubbed(finding: Finding) -> Finding:
    return finding.model_copy(
        update={
            "evidence": scrub(finding.evidence),
            "reason": scrub(finding.reason) if finding.reason else None,
        }
    )


class GuardService:
    def __init__(
        self,
        checks: DeterministicChecks,
        judge: LLMJudge,
        engine: DecisionEngine,
        audit: JsonlAuditLog,
        policy_version: str,
    ):
        self.checks = checks
        self.judge = judge
        self.engine = engine
        self.audit = audit
        self.policy_version = policy_version

    def guard(self, request: GuardRequest) -> GuardResponse:
        started = time.perf_counter()
        request_id = str(uuid4())

        deterministic = self.checks.run(request.feature_id, request.llm_output)
        judged = self.judge.review(
            request.feature_id,
            request.user_prompt,
            request.llm_output,
            context=request.context,
            deterministic=deterministic,
        )

        findings = deterministic.findings + judged.findings
        outcome = self.engine.decide(
            findings, judged, high_risk=self.judge.is_high_risk(request.feature_id)
        )
        final_output = request.llm_output if outcome.decision in RELEASABLE else None

        self.audit.record(
            AuditEntry(
                request_id=request_id,
                feature_id=request.feature_id,
                input_hash=sha256_hex(request.user_prompt),
                output_hash=sha256_hex(request.llm_output),
                policy_version=self.policy_version,
                decision=outcome.decision,
                risk_score=outcome.risk_score,
                reasons=[scrub(reason) for reason in outcome.reasons],
                findings=[_scrubbed(f) for f in findings],
                judge={
                    "skipped": judged.skipped,
                    "skip_reason": judged.skip_reason,
                    "judged_policies": judged.judged_policies,
                    "disagreements": judged.disagreements,
                    "failed_policies": {k: scrub(v) for k, v in judged.failed_policies.items()},
                    "unverified_discarded": judged.unverified_discarded,
                },
                stage_latency_ms={**deterministic.stage_latency_ms, "judge": judged.latency_ms},
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        )

        return GuardResponse(
            request_id=request_id,
            decision=outcome.decision,
            risk_score=outcome.risk_score,
            findings=findings,
            reasons=outcome.reasons,
            policy_version=self.policy_version,
            final_output=final_output,
        )


def build_service(settings: Settings) -> GuardService:
    registry = load_policies()
    try:
        client = build_judge_client(settings)
    except JudgeConfigError as exc:
        # Fail closed: every judge call fails, so high-risk features go to review
        # and low-risk ones are released with a "not fully checked" warning.
        logger.warning("LLM judge unavailable: %s", exc)
        client = UnavailableJudgeClient(str(exc))

    return GuardService(
        DeterministicChecks.from_registry(registry),
        LLMJudge.from_registry(registry, client),
        DecisionEngine(settings.low_confidence_threshold),
        JsonlAuditLog(settings.audit_log_path),
        registry.version,
    )