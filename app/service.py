"""The guardrail service: checks -> judge -> decision -> rewrite or block -> audit log."""
import json
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
from app.rewrite.block import (
    BLOCK_MESSAGE,
    DEFAULT_REASON_CODE,
    blocking_policy,
    public_findings,
    reason_code,
)
from app.rewrite.redact import PIIRedactor
from app.rewrite.rewriter import Rewriter

logger = logging.getLogger("policygate")

# Output is released only on these decisions. A REWRITE decision releases the
# rewritten text instead (see GuardService.guard); everything else returns None.
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
        rewriter: Rewriter | None = None,
    ):
        self.checks = checks
        self.judge = judge
        self.engine = engine
        self.audit = audit
        self.policy_version = policy_version
        self.rewriter = rewriter  # None: REWRITE decisions return no final_output

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

        decision = outcome.decision
        reasons = outcome.reasons
        final_output = request.llm_output if decision in RELEASABLE else None
        blocked_by: str | None = None

        if decision == Decision.REWRITE and self.rewriter is not None:
            attempt = self.rewriter.attempt(request, findings)

        if attempt is not None:
         decision = attempt.decision
        final_output = attempt.final_output
        reasons = attempt.merge_reasons(reasons)
        blocked_by = attempt.blocking_policy

        # Preserve initial findings and append new re-check findings.
        findings.extend(attempt.findings)
        if decision == Decision.BLOCK and blocked_by is None:
            blocked_by = blocking_policy(findings, self.engine.low_confidence_threshold)

        # Raw findings stay internal. Anything that leaves this method, to the
        # audit log or the client, is scrubbed of PII.
        safe_findings = [_scrubbed(f) for f in findings]
        safe_reasons = [scrub(reason) for reason in reasons]
        blocked = decision == Decision.BLOCK

        self.audit.record(
            AuditEntry(
                request_id=request_id,
                feature_id=request.feature_id,
                input_hash=sha256_hex(
                    json.dumps(
                        request.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                ),
                output_hash=sha256_hex(request.llm_output),
                policy_version=self.policy_version,
                decision=decision,
                risk_score=outcome.risk_score,
                reasons=safe_reasons,
                findings=safe_findings,
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
            decision=decision,
            risk_score=outcome.risk_score,
            findings=public_findings(safe_findings) if blocked else safe_findings,
            reasons=safe_reasons,
            policy_version=self.policy_version,
            final_output=final_output,
            reason_code=reason_code(blocked_by or DEFAULT_REASON_CODE) if blocked else None,
            message=BLOCK_MESSAGE if blocked else None,
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

    checks = DeterministicChecks.from_registry(registry)
    return GuardService(
        checks,
        LLMJudge.from_registry(registry, client),
        DecisionEngine(settings.low_confidence_threshold),
        JsonlAuditLog(settings.audit_log_path),
        registry.version,
        rewriter=Rewriter(registry, checks, PIIRedactor(registry.get("pii_leakage"))),
    )
