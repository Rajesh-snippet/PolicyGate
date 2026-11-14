import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_service
from app.service import build_service

EMAIL = "jane.doe@example.com"


@pytest.fixture
def client_and_log(tmp_path):
    log_path = tmp_path / "audit.jsonl"
    settings = Settings(audit_log_path=str(log_path), groq_api_key=None)
    service = build_service(settings)
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app), log_path
    app.dependency_overrides.clear()


def _post(client, output):
    return client.post(
        "/guard",
        json={"feature_id": "support_bot", "user_prompt": "hi", "llm_output": output},
    )


def test_response_masks_pii_in_findings_and_reasons(client_and_log):
    client, _ = client_and_log
    response = _post(client, f"Contact me at {EMAIL}")
    assert response.status_code == 200
    assert EMAIL not in response.text
    assert "[redacted:email]" in response.text


def test_masking_does_not_change_the_decision(client_and_log):
    client, _ = client_and_log
    body = _post(client, f"Contact me at {EMAIL}").json()
    assert body["decision"] == "REWRITE"
    assert body["final_output"] is None
    assert body["findings"][0]["policy"] == "pii_leakage"


def test_audit_log_never_contains_the_raw_value(client_and_log):
    client, log_path = client_and_log
    _post(client, f"Contact me at {EMAIL}")
    assert EMAIL not in log_path.read_text(encoding="utf-8")