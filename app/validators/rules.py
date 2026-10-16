import re

from app.models import Finding
from app.policies import Policy, PolicyRegistry, Strategy


class RuleEngine:
    """Applies the regex patterns from YAML policies to LLM output.

    Only deterministic and hybrid policies that define `patterns` are used.
    Detector-based policies (PII, schema) are handled by their own validators,
    and pure llm_judge policies are left to the judge.
    """

    def __init__(self, policies: list[Policy]):
        self._rules: list[tuple[Policy, list[tuple[str, re.Pattern[str]]]]] = []
        for policy in policies:
            if policy.strategy in (Strategy.DETERMINISTIC, Strategy.HYBRID) and policy.patterns:
                compiled = [(p, re.compile(p, re.IGNORECASE)) for p in policy.patterns]
                self._rules.append((policy, compiled))

    @classmethod
    def from_registry(cls, registry: PolicyRegistry) -> "RuleEngine":
        return cls(registry.all())

    def check(self, feature_id: str, llm_output: str) -> list[Finding]:
        # feature_id is part of the shared validator interface; rules are
        # global for now, but per-feature rule sets can hook in here later.
        findings: list[Finding] = []
        seen: set[tuple[str, str]] = set()

        for policy, patterns in self._rules:
            for pattern_text, pattern in patterns:
                for match in pattern.finditer(llm_output):
                    evidence = match.group()
                    if not evidence or (policy.name, evidence.lower()) in seen:
                        continue
                    seen.add((policy.name, evidence.lower()))
                    findings.append(
                        Finding(
                            policy=policy.name,
                            severity=policy.severity,
                            evidence=evidence,  # exact span, original casing
                            action=policy.action,
                            reason=f"matched rule pattern {pattern_text!r}",
                        )
                    )
        return findings