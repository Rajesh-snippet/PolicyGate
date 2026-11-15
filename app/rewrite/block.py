"""Structured BLOCK response helpers.

A blocked response tells the caller that it was blocked and gives a stable
reason_code. It never carries evidence, reason text, or policy wording; the
full (PII-scrubbed) findings stay in the audit log.
"""
import re

from app.decision import SEVERITY_RANK
from app.models import Decision, Finding

BLOCK_MESSAGE = "The response could not be provided."
DEFAULT_REASON_CODE = "POLICY_VIOLATION"
WITHHELD = "[withheld]"


def reason_code(policy: str) -> str:
    """unsafe_instruction -> UNSAFE_INSTRUCTION"""
    code = re.sub(r"[^A-Z0-9]+", "_", policy.upper()).strip("_")
    return code or DEFAULT_REASON_CODE


def blocking_policy(findings: list[Finding], low_confidence_threshold: float) -> str | None:
    """The policy that caused a BLOCK: highest severity first, then highest confidence.

    A low-confidence finding goes to a person instead of blocking, so it is not a cause.
    """
    causes = [
        f
        for f in findings
        if f.action == Decision.BLOCK
        and not (f.confidence is not None and f.confidence < low_confidence_threshold)
    ]
    if not causes:
        return None
    best = max(
        causes,
        key=lambda f: (SEVERITY_RANK[f.severity], 1.0 if f.confidence is None else f.confidence),
    )
    return best.policy


def public_findings(findings: list[Finding]) -> list[Finding]:
    """Findings safe to show a caller whose response was blocked."""
    return [f.model_copy(update={"evidence": WITHHELD, "reason": None}) for f in findings]
