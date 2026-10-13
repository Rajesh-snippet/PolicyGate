"""Policy models and YAML loader.

Policies live in policies/*.yaml (one policy per file) so rules can change
without touching engine code.
"""
import hashlib
import re
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.models import Decision, Severity

DEFAULT_POLICIES_DIR = Path(__file__).resolve().parent.parent / "policies"


class PolicyLoadError(Exception):
    """Raised when a policy file is missing, malformed, or invalid."""


class Strategy(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class PolicyExamples(BaseModel):
    bad: str
    good: str


class Policy(BaseModel):
    name: str = Field(min_length=1)
    version: str
    severity: Severity
    strategy: Strategy
    action: Decision
    fallback_action: Decision | None = None
    per_feature_override: bool = False
    description: str = ""
    patterns: list[str] = Field(default_factory=list)
    detectors: list[str] = Field(default_factory=list)
    examples: PolicyExamples | None = None

    @field_validator("action", "fallback_action", mode="before")
    @classmethod
    def _normalise_action(cls, value):
        # YAML may say "rewrite"; the enum values are upper case.
        return value.upper() if isinstance(value, str) else value

    @field_validator("patterns")
    @classmethod
    def _patterns_must_compile(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns

    @model_validator(mode="after")
    def _deterministic_needs_a_rule(self):
        if self.strategy == Strategy.DETERMINISTIC and not (self.patterns or self.detectors):
            raise ValueError("deterministic policy needs at least one pattern or detector")
        return self


class PolicyRegistry:
    def __init__(self, policies: list[Policy], version: str):
        self._policies: dict[str, Policy] = {}
        for policy in policies:
            if policy.name in self._policies:
                raise PolicyLoadError(f"duplicate policy name: {policy.name}")
            self._policies[policy.name] = policy
        self.version = version

    def get(self, name: str) -> Policy:
        return self._policies[name]

    def all(self) -> list[Policy]:
        return list(self._policies.values())

    def by_strategy(self, *strategies: Strategy) -> list[Policy]:
        return [p for p in self._policies.values() if p.strategy in strategies]

    def __len__(self) -> int:
        return len(self._policies)


def load_policies(directory: Path | str = DEFAULT_POLICIES_DIR) -> PolicyRegistry:
    directory = Path(directory)
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise PolicyLoadError(f"no policy files found in {directory}")

    policies: list[Policy] = []
    digest = hashlib.sha256()
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(raw)
        try:
            data = yaml.safe_load(raw)
            policies.append(Policy.model_validate(data))
        except (yaml.YAMLError, ValidationError) as exc:
            raise PolicyLoadError(f"{path.name}: {exc}") from exc

    # Changes whenever any policy file changes, so every decision can record it.
    return PolicyRegistry(policies, version=digest.hexdigest()[:12])
