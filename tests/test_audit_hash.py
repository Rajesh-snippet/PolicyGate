
from app.audit import sha256_hex
from app.models import GuardRequest


def test_identical_requests_produce_same_hash():
    request1 = GuardRequest(
        feature_id="health_assistant",
        user_prompt="What is diabetes?",
        llm_output="Diabetes is a condition.",
        context="Medical reference",
    )

    request2 = GuardRequest(
        feature_id="health_assistant",
        user_prompt="What is diabetes?",
        llm_output="Diabetes is a condition.",
        context="Medical reference",
    )

    hash1 = sha256_hex(
        request1.model_dump_json(exclude_none=False)
    )
    hash2 = sha256_hex(
        request2.model_dump_json(exclude_none=False)
    )

    assert hash1 == hash2


def test_different_outputs_produce_different_hashes():
    request1 = GuardRequest(
        feature_id="health_assistant",
        user_prompt="What is diabetes?",
        llm_output="Diabetes is a condition.",
    )

    request2 = request1.model_copy(
        update={"llm_output": "Diabetes affects blood sugar."}
    )

    hash1 = sha256_hex(
        request1.model_dump_json(exclude_none=False)
    )
    hash2 = sha256_hex(
        request2.model_dump_json(exclude_none=False)
    )

    assert hash1 != hash2
