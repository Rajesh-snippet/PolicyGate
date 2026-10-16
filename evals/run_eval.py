"""Evaluate the guardrail against the labeled dataset.

Usage (from the project root):
    python -m evals.run_eval

Only examples with layer == "deterministic" are scored against the
deterministic checks. "judge" examples are labeled now and scored once the
LLM judge exists (Day 6).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.models import Decision
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


def detected_policies(checks: DeterministicChecks, example: EvalExample) -> set[str]:
    result = checks.run(example.feature_id, example.llm_output)
    return {finding.policy for finding in result.findings}


def evaluate(
    examples: list[EvalExample], checks: DeterministicChecks, policy_names: list[str]
) -> dict[str, PolicyMetrics]:
    metrics = {name: PolicyMetrics() for name in policy_names}
    for example in examples:
        if example.layer != "deterministic":
            continue
        expected = set(example.expected_policies)
        found = detected_policies(checks, example)
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
    registry = load_policies()
    checks = DeterministicChecks.from_registry(registry)
    examples = load_dataset()
    names = [p.name for p in registry.all()]
    metrics = evaluate(examples, checks, names)

    scored = sum(1 for e in examples if e.layer == "deterministic")
    print(f"policy version {registry.version} | {scored} deterministic examples scored "
          f"of {len(examples)}\n")
    print(f"{'policy':<26}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'recall':>8}{'FPR':>6}")
    for name, m in metrics.items():
        print(f"{name:<26}{m.tp:>4}{m.fp:>4}{m.fn:>4}"
              f"{_fmt(m.precision):>7}{_fmt(m.recall):>8}{_fmt(m.false_positive_rate):>6}")


if __name__ == "__main__":
    main()
