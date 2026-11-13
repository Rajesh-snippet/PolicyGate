
import pytest

from app.decision import DecisionEngine
from app.judge.judge import JudgeResult
from app.models import Decision, Finding, Severity


@pytest.fixture
def engine():
    return DecisionEngine(low_confidence_threshold=0.6)


def make_finding(
    action=Decision.APPROVE_WITH_WARNING,
    severity=Severity.LOW,
    confidence=None,
):
    return Finding(
        policy="test_policy",
        severity=severity,
        evidence="test evidence",
        action=action,
        reason="Test reason",
        confidence=confidence,
    )


def test_no_findings_returns_approve(engine):
    result = engine.decide([])

    assert result.decision == Decision.APPROVE
    assert result.risk_score == 0.0


def test_low_severity_returns_warning(engine):
    finding = make_finding(
        action=Decision.APPROVE_WITH_WARNING,
        severity=Severity.LOW,
    )

    result = engine.decide([finding])

    assert result.decision == Decision.APPROVE_WITH_WARNING


def test_fixable_violation_returns_rewrite(engine):
    finding = make_finding(
        action=Decision.REWRITE,
        severity=Severity.MEDIUM,
    )

    result = engine.decide([finding])

    assert result.decision == Decision.REWRITE


def test_critical_violation_returns_block(engine):
    finding = make_finding(
        action=Decision.BLOCK,
        severity=Severity.CRITICAL,
    )

    result = engine.decide([finding])

    assert result.decision == Decision.BLOCK


def test_judge_disagreement_returns_human_review(engine):
    judge = JudgeResult(
        disagreements=["unsupported_claims"]
    )

    result = engine.decide([], judge=judge)

    assert result.decision == Decision.HUMAN_REVIEW


def test_high_risk_judge_failure_returns_human_review(engine):
    judge = JudgeResult(
        failed_policies={
            "medical_overconfidence": "API timeout"
        }
    )

    result = engine.decide(
        [],
        judge=judge,
        high_risk=True,
    )

    assert result.decision == Decision.HUMAN_REVIEW


def test_low_risk_judge_failure_returns_warning(engine):
    judge = JudgeResult(
        failed_policies={
            "unsupported_claims": "API timeout"
        }
    )

    result = engine.decide(
        [],
        judge=judge,
        high_risk=False,
    )

    assert result.decision == Decision.APPROVE_WITH_WARNING


def test_block_overrides_human_review(engine):
    finding = make_finding(
        action=Decision.BLOCK,
        severity=Severity.CRITICAL,
    )

    judge = JudgeResult(
        disagreements=["unsupported_claims"]
    )

    result = engine.decide(
        [finding],
        judge=judge,
        high_risk=True,
    )

    assert result.decision == Decision.BLOCK


def test_low_confidence_rewrite_requires_human_review(engine):
    finding = make_finding(
        action=Decision.REWRITE,
        severity=Severity.HIGH,
        confidence=0.4,
    )

    result = engine.decide([finding])

    assert result.decision == Decision.HUMAN_REVIEW


def test_high_confidence_rewrite_is_allowed(engine):
    finding = make_finding(
        action=Decision.REWRITE,
        severity=Severity.HIGH,
        confidence=0.9,
    )

    result = engine.decide([finding])

    assert result.decision == Decision.REWRITE


def test_most_severe_action_wins(engine):
    warning = make_finding(
        action=Decision.APPROVE_WITH_WARNING,
        severity=Severity.LOW,
    )

    rewrite = make_finding(
        action=Decision.REWRITE,
        severity=Severity.MEDIUM,
    )

    result = engine.decide([warning, rewrite])

    assert result.decision == Decision.REWRITE
