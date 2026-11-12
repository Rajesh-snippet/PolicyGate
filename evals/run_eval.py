"""Evaluate the guardrail against the labeled dataset.

Usage (from the project root):
    python -m evals.run_eval

By default only examples with layer == "deterministic" are scored, against the
deterministic checks. With --judge, every example is scored against the
deterministic checks plus the LLM judge (needs an API key in .env; this makes
live API calls, roughly one or two per judged policy per example):
    python -m evals.run_eval --judge
"""
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.judge.judge import JudgeResult, LLMJudge
from app.models import Decision, Finding
from app.pipeline import DeterministicChecks
from app.policies import load_policies

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


class EvalExample(BaseModel):
    id: str
    feature_id: str
    user_prompt: str
    llm_output: str
    expected_policies: list[str]
    expected_decision: Decision
    layer: Literal["deterministic", "judge"]


@dataclass
class PolicyMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else None

    @property
    def false_positive_rate(self) -> float | None:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else None


def load_dataset(path: Path | str = DEFAULT_DATASET) -> list[EvalExample]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [EvalExample.model_validate_json(line) for line in lines if line.strip()]


def review_for(
    checks: DeterministicChecks, example: EvalExample, judge: LLMJudge | None = None
) -> tuple[list[Finding], JudgeResult | None]:
    """All findings for an example, plus the judge's full result (None without a judge)."""
    result = checks.run(example.feature_id, example.llm_output)
    findings = list(result.findings)
    review = None
    if judge is not None:
        review = judge.review(
            example.feature_id,
            example.user_prompt,
            example.llm_output,
            deterministic=result,
        )
        findings += review.findings
    return findings, review


def findings_for(
    checks: DeterministicChecks, example: EvalExample, judge: LLMJudge | None = None
) -> list[Finding]:
    return review_for(checks, example, judge)[0]


def detected_policies(
    checks: DeterministicChecks, example: EvalExample, judge: LLMJudge | None = None
) -> set[str]:
    return {finding.policy for finding in findings_for(checks, example, judge)}


@dataclass
class Mismatch:
    example: EvalExample
    expected: set[str]
    found: set[str]
    findings: list[Finding]
    judge: JudgeResult | None = None

    @property
    def false_positives(self) -> set[str]:
        return self.found - self.expected

    @property
    def false_negatives(self) -> set[str]:
        return self.expected - self.found


@dataclass
class RunStats:
    """Tells a real judge miss apart from a judge that never got to answer."""

    judged_examples: int = 0  # the judge actually ran (was not skipped)
    failed_policy_calls: int = 0
    examples_with_failures: int = 0
    unverified_discarded: int = 0  # findings dropped: quote not found in the output
    disagreements: int = 0
    failure_details: list[dict] = field(default_factory=list)


def format_judge_stats(stats: RunStats) -> str:
    return (
        f"judge: ran on {stats.judged_examples} examples | "
        f"{stats.failed_policy_calls} policy calls failed (in {stats.examples_with_failures} examples) | "
        f"{stats.unverified_discarded} findings discarded (quote not in output) | "
        f"{stats.disagreements} pass disagreements"
    )


def evaluate(
    examples: list[EvalExample],
    checks: DeterministicChecks,
    policy_names: list[str],
    judge: LLMJudge | None = None,
    mismatches: list[Mismatch] | None = None,
    stats: RunStats | None = None,
    pause: float = 0.0,
) -> dict[str, PolicyMetrics]:
    """Score examples.

    mismatches: if given, wrongly flagged or missed examples are added to it.
    stats: if given, judge failures and discards are counted in it.
    pause: seconds to wait after each example the judge ran on (free-tier rate limits).
    """
    metrics = {name: PolicyMetrics() for name in policy_names}
    for example in examples:
        if judge is None and example.layer != "deterministic":
            continue
        expected = set(example.expected_policies)
        findings, review = review_for(checks, example, judge)
        found = {finding.policy for finding in findings}
        if mismatches is not None and found != expected:
            mismatches.append(Mismatch(example, expected, found, findings, review))
        if review is not None and not review.skipped:
            if stats is not None:
                stats.judged_examples += 1
                for policy, error in review.failed_policies.items():
                      stats.failure_details.append({
                     "example_id": example.id,
                     "feature_id": example.feature_id,
                     "policy": policy,
                     "error": error,
                       })
                stats.failed_policy_calls += len(review.failed_policies)
                stats.examples_with_failures += 1 if review.failed else 0
                stats.unverified_discarded += review.unverified_discarded
                stats.disagreements += len(review.disagreements)
            if pause:
                time.sleep(pause)
        for name, m in metrics.items():
            if name in expected and name in found:
                m.tp += 1
            elif name in found:
                m.fp += 1
            elif name in expected:
                m.fn += 1
            else:
                m.tn += 1
    return metrics


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="also score with the LLM judge")
    parser.add_argument(
        "--verbose", action="store_true", help="list each wrongly flagged or missed example"
    )
    parser.add_argument(
        "--pause", type=float, default=0.0,
        help="seconds to wait between judged examples (helps with free-tier rate limits)",
    )
    args = parser.parse_args()

    registry = load_policies()
    checks = DeterministicChecks.from_registry(registry)
    examples = load_dataset()
    names = [p.name for p in registry.all()]

    judge = None
    if args.judge:
        from app.config import get_settings
        from app.judge.client import build_judge_client

        judge = LLMJudge.from_registry(registry, build_judge_client(get_settings()))
    mismatches: list[Mismatch] = []
    stats = RunStats()
    metrics = evaluate(examples, checks, names, judge, mismatches, stats, args.pause)

    scored = len(examples) if judge else sum(1 for e in examples if e.layer == "deterministic")
    mode = "deterministic + judge" if judge else "deterministic only"
    print(f"policy version {registry.version} | {mode} | {scored} of {len(examples)} examples scored\n")
    print(f"{'policy':<26}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'recall':>8}{'FPR':>6}")
    for name, m in metrics.items():
        print(f"{name:<26}{m.tp:>4}{m.fp:>4}{m.fn:>4}"
              f"{_fmt(m.precision):>7}{_fmt(m.recall):>8}{_fmt(m.false_positive_rate):>6}")
    if judge:
        print(f"\n{format_judge_stats(stats)}")
    if stats.failure_details:
        print("\nFailed judge calls:")
        for failure in stats.failure_details:
            print(
                f"- Example: {failure['example_id']} | "
                f"Feature: {failure['feature_id']} | "
                f"Policy: {failure['policy']}"
            )
            print(f"  Error: {failure['error']}")
    else:
        print("\nFailed judge calls: none")

    if args.verbose:
        print(f"\n{len(mismatches)} example(s) with wrong or missed findings:")
        for item in mismatches:
            print(f"\n[{item.example.id}] feature={item.example.feature_id}")
            print(f"  output:   {item.example.llm_output}")
            print(f"  expected: {sorted(item.expected) or 'no findings'}")
            if item.false_positives:
                print(f"  wrongly flagged: {sorted(item.false_positives)}")
            if item.false_negatives:
                print(f"  missed:          {sorted(item.false_negatives)}")
            review = item.judge
            if review is not None:
                if review.skipped:
                    print(f"  judge skipped: {review.skip_reason}")
                else:
                    print(f"  judge asked about: {review.judged_policies}")
                if review.unverified_discarded:
                    print(f"  judge findings discarded (quote not in output): {review.unverified_discarded}")
                if review.disagreements:
                    print(f"  judge passes disagreed on: {review.disagreements}")
            for finding in item.findings:
                if finding.policy in item.false_positives:
                    confidence = "-" if finding.confidence is None else f"{finding.confidence:.2f}"
                    print(f"    - {finding.policy} (confidence {confidence}): "
                          f"quoted {finding.evidence!r}; reason: {finding.reason}")


if __name__ == "__main__":
    main()