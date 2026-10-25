"""Turns findings into one explainable decision.

Precedence, highest wins:
    BLOCK > HUMAN_REVIEW > REWRITE > APPROVE_WITH_WARNING > APPROVE

- Each finding votes with its policy's action.
- A judge finding below the confidence threshold cannot trigger an automatic
  REWRITE or BLOCK; it goes to a person. Deterministic findings are certain.
- Judge disagreement goes to HUMAN_REVIEW.
- A judge failure on a high-risk feature fails closed to HUMAN_REVIEW. On a
  low-risk feature the output is released with a warning that it was not fully
  checked.
- risk_score is a reporting signal only; it never changes the decision.
"""
from dataclasses import dataclass, field

from app.judge.judge import JudgeResult
from app.models import Decision, Finding, Severity

ACTION_RANK = {
    Decision.APPROVE: 0,
    Decision.APPROVE_WITH_WARNING: 1,
    Decision.REWRITE: 2,
    Decision.HUMAN_REVIEW: 3,
    Decision.BLOCK: 4,
}

SEVERITY_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}

SEVERITY_WEIGHT = {
    Severity.LOW: 0.2,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.0,
}

UNCERTAIN_RISK_FLOOR = 0.5


@dataclass
class DecisionOutcome:
    decision: Decision
    risk_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class _Vote:
    decision: Decision
    severity_rank: int
    reason: str


class DecisionEngine:
    def __init__(self, low_confidence_threshold: float = 0.6):
        self.low_confidence_threshold = low_confidence_threshold

    def decide(
        self,
        findings: list[Finding],
        judge: JudgeResult | None = None,
        high_risk: bool = False,
    ) -> DecisionOutcome:
        votes = [self._vote(f) for f in findings]
        uncertain = False

        if judge is not None:
            for policy in judge.disagreements:
                votes.append(_Vote(Decision.HUMAN_REVIEW, 0, f"judge disagreement on {policy} -> HUMAN_REVIEW"))
                uncertain = True
            for policy in judge.failed_policies:
                decision = Decision.HUMAN_REVIEW if high_risk else Decision.APPROVE_WITH_WARNING
                votes.append(
                    _Vote(decision, 0, f"{policy} not fully checked (judge failed) -> {decision.value}")
                )
                uncertain = True

        if not votes:
            return DecisionOutcome(Decision.APPROVE, 0.0, ["no findings"])

        votes.sort(key=lambda v: (ACTION_RANK[v.decision], v.severity_rank), reverse=True)
        decision = votes[0].decision
        return DecisionOutcome(
            decision, self._risk_score(findings, uncertain), [v.reason for v in votes]
        )

    def _vote(self, finding: Finding) -> _Vote:
        severity_rank = SEVERITY_RANK[finding.severity]
        label = f"{finding.policy} ({finding.severity.value})"
        low_confidence = (
            finding.confidence is not None and finding.confidence < self.low_confidence_threshold
        )
        if low_confidence and finding.action in (Decision.REWRITE, Decision.BLOCK):
            return _Vote(
                Decision.HUMAN_REVIEW,
                severity_rank,
                f"{label} low confidence ({finding.confidence:.2f}) -> HUMAN_REVIEW",
            )
        return _Vote(finding.action, severity_rank, f"{label} -> {finding.action.value}")

    @staticmethod
    def _risk_score(findings: list[Finding], uncertain: bool) -> float:
        score = max(
            (
                SEVERITY_WEIGHT[f.severity] * (1.0 if f.confidence is None else f.confidence)
                for f in findings
            ),
            default=0.0,
        )
        if uncertain:
            score = max(score, UNCERTAIN_RISK_FLOOR)
        return round(score, 2)
