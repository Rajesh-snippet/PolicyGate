"""Unit tests for Day 8a/8b: PII redaction, BLOCK helpers, and the Rewriter.

These use fake checks, so they need no network, no judge and no policy files.
"""
from types import SimpleNamespace

import pytest

from app.audit import scrub
from app.models import Decision, Finding, GuardRequest, Severity
from app.policies import Policy, PolicyRegistry
from app.rewrite.block import (
    BLOCK_MESSAGE,
    WITHHELD,
    blocking_policy,
    public_findings,
    reason_code,
)
from app.rewrite.redact import PIIRedactor
from app.rewrite.rewriter import RewriteAttempt, Rewriter
from app.validators.pii import PIIDetector

EMAIL = "jane.doe@example.com"
PHONE = "415-555-0142"
KEY = "sk-abcdefghijklmnopqrstuvwxyz123456"
CARD = "4111 1111 1111 1111"


def _policy(name="pii_leakage", fallback="block", **overrides) -> Policy:
    data = {
        "name": name,
        "version": "1.0.0",
        "severity": "high",
        "strategy": "deterministic",
        "action": "rewrite",
        "fallback_action": fallback,
        "per_feature_override": True,
        "detectors": ["email", "phone", "credit_card", "api_key", "ip_address"],
    }
    data.update(overrides)
    return Policy.model_validate(data)


def _finding(policy, action, severity=Severity.HIGH, confidence=None) -> Finding:
    return Finding(
        policy=policy, severity=severity, evidence="x", action=action, confidence=confidence
    )


def _request(output, feature="support_bot") -> GuardRequest:
    return GuardRequest(feature_id=feature, user_prompt="hi", llm_output=output)


class _PIIChecks:
    """Behaves like the deterministic pipeline for the PII policy only."""

    def __init__(self, policy):
        self.detector = PIIDetector(policy)

    def run(self, feature_id, output):
        return SimpleNamespace(findings=self.detector.detect(feature_id, output))


class _FixedChecks:
    def __init__(self, findings):
        self.findings = findings

    def run(self, feature_id, output):
        return SimpleNamespace(findings=self.findings)


# ---- redaction ---------------------------------------------------------------


@pytest.fixture
def redactor():
    return PIIRedactor(_policy())


def test_redacts_each_kind_with_audit_style_tokens(redactor):
    text = f"Mail {EMAIL}, call {PHONE}, key {KEY}, card {CARD}."
    out = redactor.redact("support_bot", text)
    assert out == (
        "Mail [redacted:email], call [redacted:phone], "
        "key [redacted:api_key], card [redacted:credit_card]."
    )


def test_redaction_matches_the_audit_scrubber(redactor):
    text = f"Mail {EMAIL}, call {PHONE}, key {KEY}, card {CARD}."
    assert redactor.redact("support_bot", text) == scrub(text)


def test_redacted_text_has_no_pii_left(redactor):
    detector = PIIDetector(_policy())
    text = f"Mail {EMAIL}, call {PHONE}, key {KEY}, card {CARD}, host 10.0.0.12"
    assert detector.detect("support_bot", redactor.redact("support_bot", text)) == []


def test_secret_assignment_is_redacted_and_not_redetected(redactor):
    out = redactor.redact("support_bot", "password = hunter2hunter2hunter2")
    assert "hunter2" not in out
    assert PIIDetector(_policy()).detect("support_bot", out) == []


def test_clean_text_is_unchanged(redactor):
    text = "Your order ships tomorrow."
    assert redactor.redact("support_bot", text) == text


def test_allowed_email_survives_but_other_pii_does_not(redactor):
    out = redactor.redact("support_assistant", f"Write to {EMAIL} or call {PHONE}")
    assert EMAIL in out
    assert "[redacted:phone]" in out


def test_never_allowed_kinds_are_redacted_even_when_email_is_allowed(redactor):
    out = redactor.redact("support_assistant", f"{EMAIL} {KEY}")
    assert EMAIL in out
    assert "[redacted:api_key]" in out


def test_unknown_detector_in_policy_is_rejected():
    with pytest.raises(ValueError):
        PIIRedactor(_policy(detectors=["email", "json_schema"]))


# ---- block helpers -----------------------------------------------------------


def test_reason_code_formats_policy_names():
    assert reason_code("unsafe_instruction") == "UNSAFE_INSTRUCTION"
    assert reason_code("pii-leakage v2") == "PII_LEAKAGE_V2"
    assert reason_code("!!!") == "POLICY_VIOLATION"


def test_blocking_policy_prefers_higher_severity():
    findings = [
        _finding("medical_overconfidence", Decision.BLOCK, Severity.HIGH),
        _finding("unsafe_instruction", Decision.BLOCK, Severity.CRITICAL),
        _finding("brand_tone", Decision.APPROVE_WITH_WARNING, Severity.LOW),
    ]
    assert blocking_policy(findings, 0.6) == "unsafe_instruction"


def test_blocking_policy_ignores_low_confidence_findings():
    findings = [
        _finding("unsafe_instruction", Decision.BLOCK, Severity.CRITICAL, confidence=0.3),
        _finding("medical_overconfidence", Decision.BLOCK, Severity.HIGH, confidence=0.9),
    ]
    assert blocking_policy(findings, 0.6) == "medical_overconfidence"


def test_blocking_policy_is_none_without_a_block_finding():
    assert blocking_policy([_finding("pii_leakage", Decision.REWRITE)], 0.6) is None


def test_public_findings_withhold_evidence_and_reason():
    original = Finding(
        policy="unsafe_instruction",
        severity=Severity.CRITICAL,
        evidence="step-by-step dangerous text",
        action=Decision.BLOCK,
        reason="explains how to cause harm",
        confidence=0.95,
    )
    [shown] = public_findings([original])
    assert shown.policy == "unsafe_instruction"
    assert shown.severity == Severity.CRITICAL
    assert shown.action == Decision.BLOCK
    assert shown.evidence == WITHHELD
    assert shown.reason is None
    assert original.evidence == "step-by-step dangerous text"  # original untouched


def test_block_message_is_fixed():
    assert BLOCK_MESSAGE == "The response could not be provided."


# ---- rewriter ----------------------------------------------------------------


def _rewriter(checks=None, fallback="block"):
    policy = _policy(fallback=fallback)
    registry = PolicyRegistry(
        [policy, _policy("unsupported_claims", fallback="human_review", detectors=[], patterns=["x"])],
        "v",
    )
    return Rewriter(registry, checks or _PIIChecks(policy), PIIRedactor(policy))


def test_pii_only_rewrite_returns_redacted_output():
    out = _rewriter().attempt(
        _request(f"Contact me at {EMAIL}"), [_finding("pii_leakage", Decision.REWRITE)]
    )
    assert out.decision == Decision.REWRITE
    assert out.final_output == "Contact me at [redacted:email]"
    assert out.blocking_policy is None


def test_other_rewrite_policies_are_not_handled():
    findings = [
        _finding("pii_leakage", Decision.REWRITE),
        _finding("unsupported_claims", Decision.REWRITE),
    ]
    assert _rewriter().attempt(_request(f"{EMAIL} is 100% accurate"), findings) is None


def test_non_pii_rewrite_is_not_handled():
    findings = [_finding("unsupported_claims", Decision.REWRITE)]
    assert _rewriter().attempt(_request("anything"), findings) is None


def test_warning_findings_do_not_stop_a_pii_rewrite():
    findings = [
        _finding("pii_leakage", Decision.REWRITE),
        _finding("brand_tone", Decision.APPROVE_WITH_WARNING, Severity.LOW),
    ]
    out = _rewriter().attempt(_request(f"Mail {EMAIL}"), findings)
    assert out.decision == Decision.REWRITE


def test_failed_recheck_uses_the_policy_fallback_block():
    still_bad = _FixedChecks([_finding("pii_leakage", Decision.REWRITE)])
    out = _rewriter(still_bad, fallback="block").attempt(
        _request(f"Mail {EMAIL}"), [_finding("pii_leakage", Decision.REWRITE)]
    )
    assert out.decision == Decision.BLOCK
    assert out.final_output is None
    assert out.blocking_policy == "pii_leakage"
    assert out.reasons == ["pii_leakage still failing after rewrite -> BLOCK"]


def test_failed_recheck_with_human_review_fallback():
    still_bad = _FixedChecks([_finding("pii_leakage", Decision.REWRITE)])
    out = _rewriter(still_bad, fallback="human_review").attempt(
        _request(f"Mail {EMAIL}"), [_finding("pii_leakage", Decision.REWRITE)]
    )
    assert out.decision == Decision.HUMAN_REVIEW
    assert out.blocking_policy is None


def test_missing_or_invalid_fallback_goes_to_human_review():
    still_bad = _FixedChecks([_finding("pii_leakage", Decision.REWRITE)])
    for fallback in (None, "approve"):
        out = _rewriter(still_bad, fallback=fallback).attempt(
            _request(f"Mail {EMAIL}"), [_finding("pii_leakage", Decision.REWRITE)]
        )
        assert out.decision == Decision.HUMAN_REVIEW


def test_new_blocking_finding_in_recheck_blocks():
    worse = _FixedChecks([
        _finding("unsafe_instruction", Decision.BLOCK, Severity.CRITICAL)
    ])

    out = _rewriter(worse).attempt(
        _request(f"Mail {EMAIL}"),
        [_finding("pii_leakage", Decision.REWRITE)],
    )

    assert out.decision == Decision.BLOCK
    assert out.blocking_policy == "unsafe_instruction"
    assert out.final_output is None

    # Re-check findings must be returned.
    assert len(out.findings) == 1
    assert out.findings[0].policy == "unsafe_instruction"


def test_warning_in_recheck_does_not_fail_the_rewrite():
    warn = _FixedChecks([_finding("brand_tone", Decision.APPROVE_WITH_WARNING, Severity.LOW)])
    out = _rewriter(warn).attempt(
        _request(f"Mail {EMAIL}"), [_finding("pii_leakage", Decision.REWRITE)]
    )
    assert out.decision == Decision.REWRITE


def test_merge_reasons_puts_the_deciding_reason_first():
    base = ["pii_leakage (high) -> REWRITE"]
    ok = RewriteAttempt(Decision.REWRITE, "x", ["pii_leakage redacted, re-check clean"])
    failed = RewriteAttempt(Decision.BLOCK, None, ["pii_leakage still failing after rewrite -> BLOCK"])
    assert ok.merge_reasons(base) == base + ok.reasons
    assert failed.merge_reasons(base) == failed.reasons + base
