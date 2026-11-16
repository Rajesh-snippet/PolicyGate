"""REWRITE flow. Day 8a: deterministic PII redaction plus one re-check.

attempt() returns None when it does not handle the findings, so the caller keeps
the plain REWRITE decision with no final_output. Later steps extend this class:
8c LLM rewrite for unsupported_claims and medical_overconfidence, 8d schema repair.

If the re-check still fails, the failing policy's fallback_action decides the
outcome: BLOCK or HUMAN_REVIEW (HUMAN_REVIEW when no valid fallback is set).
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.decision import ACTION_RANK
from app.models import Decision, Finding, GuardRequest
from app.policies import PolicyRegistry
from app.rewrite.redact import PIIRedactor

NEEDS_ESCALATION = {Decision.REWRITE, Decision.BLOCK, Decision.HUMAN_REVIEW}
VALID_FALLBACKS = {Decision.BLOCK, Decision.HUMAN_REVIEW}


class Checks(Protocol):
    def run(self, feature_id: str, output: str) -> Any: ...  # result has .findings


@dataclass
class RewriteAttempt:
    decision: Decision
    final_output: str | None
    reasons: list[str] = field(default_factory=list)
    blocking_policy: str | None = None
    findings: list[Finding] = field(default_factory=list)

    def merge_reasons(self, base: list[str]) -> list[str]:
        """Keep reasons ordered with the deciding one first."""
        if self.decision == Decision.REWRITE:
            return base + self.reasons
        return self.reasons + base


class Rewriter:
    def __init__(self, registry: PolicyRegistry, checks: Checks, redactor: PIIRedactor):
        self.registry = registry
        self.checks = checks
        self.redactor = redactor

    def attempt(self, request: GuardRequest, findings: list[Finding]) -> RewriteAttempt | None:
        pii = self.redactor.policy.name
        if {f.policy for f in findings if f.action == Decision.REWRITE} != {pii}:
            return None  # other policies need an LLM rewrite or schema repair (8c, 8d)
        redacted = self.redactor.redact(request.feature_id, request.llm_output)
        rechecked = self.checks.run(request.feature_id, redacted).findings

        unresolved = [f for f in rechecked if f.action in NEEDS_ESCALATION]

        if unresolved:
           return self._fall_back(unresolved, rechecked)

        return RewriteAttempt(
         Decision.REWRITE,
           redacted,
          [f"{pii} redacted, re-check clean"],
          findings=rechecked,
        )

    def _fall_back(self, unresolved: list[Finding]) -> RewriteAttempt:
        steps = [(f.policy, self._step_for(f)) for f in unresolved]
        policy, decision = max(steps, key=lambda s: ACTION_RANK[s[1]])
        reasons = [
            f"{name} still failing after rewrite -> {step.value}"
            for name, step in sorted(set(steps), key=lambda s: ACTION_RANK[s[1]], reverse=True)
        ]
        return RewriteAttempt(
            decision,
            None,
            reasons,
            blocking_policy=policy if decision == Decision.BLOCK else None,
        )

    def _step_for(self, finding: Finding) -> Decision:
        if finding.action in VALID_FALLBACKS:
            return finding.action
        try:
            fallback = self.registry.get(finding.policy).fallback_action
        except KeyError:
            fallback = None
        return fallback if fallback in VALID_FALLBACKS else Decision.HUMAN_REVIEW
