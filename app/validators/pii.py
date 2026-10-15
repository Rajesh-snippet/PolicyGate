"""Deterministic PII detection.

Each detector takes the output text and returns the matched strings.
Which detectors run comes from the `pii_leakage` policy; which are allowed
for a given feature comes from FEATURE_PII_ALLOWED.
"""
import re
from collections.abc import Callable

from app.models import Finding
from app.policies import Policy

# Detectors that no feature may allow, whatever the override says.
NEVER_ALLOWED = {"api_key", "credit_card"}

# feature_id -> detectors that are acceptable for that feature.
FEATURE_PII_ALLOWED: dict[str, set[str]] = {
    "support_assistant": {"email"},  # may legitimately show an account email
}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

PHONE_RES = [
    # international: +91 98765 43210, +1 415 555 0142
    re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?(?:\(?\d{1,5}\)?[\s.-]?){1,4}\d{2,4}(?!\d)"),
    # 10-digit with separators: (415) 555-0142, 415-555-0142
    re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?!\d)"),
    # 7-digit local: 555-0142 (loose; may match some ID formats)
    re.compile(r"(?<![\d-])\d{3}-\d{4}(?![\d-])"),
    # plain 10-digit mobile number: 9876543210
    re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
]

CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

API_KEY_RES = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
]
# api_key = "...", token: ..., password=...  (the value is the evidence)
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|secret|access[_-]?token|token|password)\b\s*[:=]\s*[\"']?"
    r"([A-Za-z0-9_\-./+]{16,})[\"']?",
    re.IGNORECASE,
)

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.])(?:{_OCTET}\.){{3}}{_OCTET}(?!\d)(?!\.\d)")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


def detect_email(text: str) -> list[str]:
    return EMAIL_RE.findall(text)


def detect_phone(text: str) -> list[str]:
    spans: list[tuple[int, int]] = []
    for pattern in PHONE_RES:
        for match in pattern.finditer(text):
            digits = re.sub(r"\D", "", match.group())
            if 7 <= len(digits) <= 15:
                spans.append((match.start(), match.end()))

    # Keep the longest matches and drop any that overlap them.
    kept: list[tuple[int, int]] = []
    for start, end in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        if all(end <= k_start or start >= k_end for k_start, k_end in kept):
            kept.append((start, end))
    return [text[start:end] for start, end in sorted(kept)]


def detect_credit_card(text: str) -> list[str]:
    found = []
    for match in CARD_CANDIDATE_RE.finditer(text):
        candidate = match.group().strip()
        digits = re.sub(r"\D", "", candidate)
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            found.append(candidate)
    return found


def detect_api_key(text: str) -> list[str]:
    found = []
    for pattern in API_KEY_RES:
        found.extend(match.group() for match in pattern.finditer(text))
    found.extend(match.group(1) for match in SECRET_ASSIGNMENT_RE.finditer(text))
    return found


def detect_ip_address(text: str) -> list[str]:
    return [match.group() for match in IPV4_RE.finditer(text)]


DETECTORS: dict[str, Callable[[str], list[str]]] = {
    "email": detect_email,
    "phone": detect_phone,
    "credit_card": detect_credit_card,
    "api_key": detect_api_key,
    "ip_address": detect_ip_address,
}


class PIIDetector:
    """Runs the detectors named in the policy and returns structured findings."""

    def __init__(self, policy: Policy, feature_allowed: dict[str, set[str]] | None = None):
        unknown = set(policy.detectors) - set(DETECTORS)
        if unknown:
            raise ValueError(f"unknown PII detectors in policy {policy.name}: {sorted(unknown)}")
        self.policy = policy
        self.feature_allowed = FEATURE_PII_ALLOWED if feature_allowed is None else feature_allowed

    def _allowed_for(self, feature_id: str) -> set[str]:
        if not self.policy.per_feature_override:
            return set()
        return self.feature_allowed.get(feature_id, set()) - NEVER_ALLOWED

    def detect(self, feature_id: str, llm_output: str) -> list[Finding]:
        allowed = self._allowed_for(feature_id)
        findings: list[Finding] = []
        seen: set[tuple[str, str]] = set()

        for detector_name in self.policy.detectors:
            if detector_name in allowed:
                continue
            for match in DETECTORS[detector_name](llm_output):
                if (detector_name, match) in seen:
                    continue
                seen.add((detector_name, match))
                findings.append(
                    Finding(
                        policy=self.policy.name,
                        severity=self.policy.severity,
                        evidence=match,
                        action=self.policy.action,
                        reason=f"{detector_name} detected",
                    )
                )
        return findings