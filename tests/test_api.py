from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_guard_returns_valid_response():
    payload = {
        "feature_id": "health_assistant",
        "user_prompt": "Give me medical advice",
        "llm_output": "This medicine will definitely cure you.",
    }
    response = client.post("/guard", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in {
        "APPROVE",
        "APPROVE_WITH_WARNING",
        "REWRITE",
        "BLOCK",
        "HUMAN_REVIEW",
    }
    assert 0 <= body["risk_score"] <= 1
    assert body["request_id"]


def test_guard_rejects_missing_fields():
    response = client.post("/guard", json={"user_prompt": "hi"})
    assert response.status_code == 422
