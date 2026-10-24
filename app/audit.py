"""Basic audit log: one JSON line per decision.

Day 9 moves this to PostgreSQL with the full field set. Raw prompts and
outputs are never stored (only hashes), and PII is scrubbed from anything
text-like, so the log itself is not a place where sensitive data leaks.
"""
import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.models import Decision, Finding
from app.validators.pii import DETECTORS

# Most specific first, so a key or card number is not first read as a phone number.
SCRUB_ORDER = ["api_key", "credit_card", "email", "ip_address", "phone"]


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scrub(text: str) -> str:
    """Replace detected PII with tokens like [redacted:email]."""
    for name in SCRUB_ORDER:
        for match in sorted(set(DETECTORS[name](text)), key=len, reverse=True):
            text = text.replace(match, f"[redacted:{name}]")
    return text


class AuditEntry(BaseModel):
    request_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    feature_id: str
    input_hash: str
    output_hash: str
    policy_version: str
    decision: Decision
    risk_score: float
    reasons: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    judge: dict[str, Any] = Field(default_factory=dict)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0


class JsonlAuditLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: AuditEntry) -> None:
        line = entry.model_dump_json()
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def read_all(self) -> list[AuditEntry]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [AuditEntry.model_validate_json(line) for line in lines if line.strip()]
