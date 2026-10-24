"""The guardrail service: deterministic checks -> judge -> decision -> audit log."""
import logging
import time
from datetime import datetime, timezone
from uuid import uuid4

from app.audit import AuditLogger, JsonlAuditLogger, audit_finding, sha256_text
from app.config import Settings
from app.decision import DecisionEngine
from app.judge.client import JudgeConfigError, build_judge_client
from app.judge.judge import JudgeResult, LLMJudge
from app.models import Decision, GuardRequest, GuardResponse
from app.pipeline import DeterministicChecks
from app.policies import load_policies

logger = logging.getLogger("policygate")

# Until the rewrite and block flows exist (Day 8), only approved output is returned.
RELEASABLE = {Decision.APPROVE, Decision.APPROVE_WITH_WARNING}


class GuardService:
    def __init__(
        self,
        checks: DeterministicChecks,
        engine: DecisionEngine,
        audit: AuditLogger,
        policy_version: str,
        judge: LLMJudge | None = None,
    ):
        self.checks = checks
        self.engine = engine
        self.audit = audit
        self.policy_version = policy_version
        self.judge = judge

    def guard(self, request: GuardRequest) -> GuardResponse:
        started = time.perf_counter()
        request_id = str(uuid4())

        deterministic = self.checks.run(request.feature_id, request.llm_output)
        if self.judge is not None:
            judge_result = self.judge.review(
                request.feature_id,
                request.user_prompt,
                request.llm_output,
                context=request.context,
                deterministic=deterministic,
            )
            high_risk = self.judge.is_high_risk(request.feature_id)
        else:
            judge_result = JudgeResult(skipped=True, skip_reason="judge not configured")
            high_risk = False

        findings = deterministic.findings + judge_result.findings
        outcome = self.engine.decide(findings, judge_result, high_risk)
        reasons = list(outcome.reasons)
        if self.judge is None:
            reasons.append("JUDGE_NOT_CONFIGURED")

        final_output = request.llm_output if outcome.decision in RELEASABLE else None
        total_ms = round((time.perf_counter() - started) * 1000, 3)

        self.audit.log(
            {
                "request_id": request_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "feature_id": request.feature_id,
                "input_hash": sha256_text(request.user_prompt),
                "output_hash": sha256_text(request.llm_output),
                "policy_version": self.policy_version,
                "decision": outcome.decision.value,
                "risk_score": outcome.risk_score,
                "reasons": reasons,
                "findings": [audit_finding(f) for f in findings],
                "judge": {
                    "skipped": judge_result.skipped,
                    "skip_reason": judge_result.skip_reason,
                    "judged_policies": judge_result.judged_policies,
                    "disagreements": judge_result.disagreements,
                    "failed_policies": judge_result.failed_policies,
                    "unverified_discarded": judge_result.unverified_discarded,
                },
                "latency_ms": {
                    "total": total_ms,
                    **deterministic.stage_latency_ms,
                    "judge": judge_result.latency_ms,
                },
            }
        )

        return GuardResponse(
            request_id=request_id,
            decision=outcome.decision,
            risk_score=outcome.risk_score,
            findings=findings,
            reasons=reasons,
            policy_version=self.policy_version,
            final_output=final_output,
        )


def build_service(settings: Settings) -> GuardService:
    registry = load_policies()
    checks = DeterministicChecks.from_registry(registry)

    judge = None
    if settings.judge_enabled:
        try:
            judge = LLMJudge.from_registry(registry, build_judge_client(settings))
        except JudgeConfigError as exc:
            logger.warning("LLM judge disabled: %s", exc)

    return GuardService(
        checks=checks,
        engine=DecisionEngine(),
        audit=JsonlAuditLogger(settings.audit_log_path),
        policy_version=registry.version,
        judge=judge,
    )