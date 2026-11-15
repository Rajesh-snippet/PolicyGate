"""End-to-end checks for Day 8a/8b through the real service and policy files.

No judge key is needed: every case below is decided by deterministic findings,
and the judge is skipped on a hard block and on low-risk features that already
have deterministic findings.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_service
from app.service import build_service

EMAIL = "jane.doe@example.com"
DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"


def _deterministic_block_examples():
    if not DATASET.exists():
        return []
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        pytest.param(row, id=row["id"])
        for row in rows
        if row.get("layer") == "deterministic" and row.get("expected_decision") == "BLOCK"
    ]


@pytest.fixture
def client_and_log(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    settings = Settings(audit_log_path=str(log_path), groq_api_key=None)
    service = build_service(settings)
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app), log_path
    app.dependency_overrides.clear()


def _post(client, output, feature="support_bot"):
    return client.post(
        "/guard", json={"feature_id": feature, "user_prompt": "hi", "llm_output": output}
    )


def test_email_is_redacted_and_released_as_a_rewrite(client_and_log):
    client, _ = client_and_log
    body = _post(client, f"Contact me at {EMAIL}").json()
    assert body["decision"] == "REWRITE"
    assert body["final_output"] == "Contact me at [redacted:email]"
    assert body["reason_code"] is None and body["message"] is None
    assert EMAIL not in json.dumps(body)


def test_allowed_email_is_kept_but_phone_is_redacted(client_and_log):
    client, _ = client_and_log
    body = _post(
        client, f"Write to {EMAIL} or call 415-555-0142", feature="support_assistant"
    ).json()
    assert body["decision"] == "REWRITE"
    assert body["final_output"] == f"Write to {EMAIL} or call [redacted:phone]"


def test_api_key_is_redacted(client_and_log):
    client, _ = client_and_log
    body = _post(client, "Use sk-abcdefghijklmnopqrstuvwxyz123456 to log in").json()
    assert body["decision"] == "REWRITE"
    assert body["final_output"] == "Use [redacted:api_key] to log in"


def test_clean_output_is_released_unchanged(client_and_log):
    client, _ = client_and_log
    body = _post(client, "Your order ships tomorrow.").json()
    assert body["decision"] in {"APPROVE", "APPROVE_WITH_WARNING"}
    assert body["final_output"] == "Your order ships tomorrow."


@pytest.mark.parametrize("row", _deterministic_block_examples())
def test_block_response_is_structured_and_withholds_details(client_and_log, row):
    client, log_path = client_and_log
    body = client.post(
        "/guard",
        json={
            "feature_id": row["feature_id"],
            "user_prompt": row["user_prompt"],
            "llm_output": row["llm_output"],
        },
    ).json()
    assert body["decision"] == "BLOCK"
    assert body["final_output"] is None
    assert body["message"] == "The response could not be provided."
    assert body["reason_code"] and body["reason_code"] == body["reason_code"].upper()
    assert body["findings"]
    assert all(f["evidence"] == "[withheld]" and f["reason"] is None for f in body["findings"])
    assert row["llm_output"] not in json.dumps(body)
    assert '"decision":"BLOCK"' in log_path.read_text(encoding="utf-8")
