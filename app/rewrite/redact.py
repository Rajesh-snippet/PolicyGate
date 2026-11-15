"""Deterministic PII redaction: the REWRITE step for pii_leakage.

Only detectors that apply to the feature are redacted, so an allowed email on
`support_assistant` survives while a phone number next to it does not. Tokens
match the audit log: [redacted:<detector>].
"""
from app.audit import SCRUB_ORDER
from app.policies import Policy
from app.validators.pii import DETECTORS, PIIDetector


class PIIRedactor:
    def __init__(self, policy: Policy, detector: PIIDetector | None = None):
        unsupported = set(policy.detectors) - set(SCRUB_ORDER)
        if unsupported:
            raise ValueError(f"cannot redact detectors: {sorted(unsupported)}")
        self.policy = policy
        self.detector = detector or PIIDetector(policy)

    def redact(self, feature_id: str, text: str) -> str:
        active = set(self.detector.active_detectors(feature_id))
        # Most specific first, so a key or card number is not first read as a phone number.
        for name in SCRUB_ORDER:
            if name not in active:
                continue
            for match in sorted(set(DETECTORS[name](text)), key=len, reverse=True):
                text = text.replace(match, f"[redacted:{name}]")
        return text
