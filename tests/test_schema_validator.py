import pytest

from app.models import Decision, Severity
from app.policies import load_policies
from app.validators.schema import SchemaValidator

FEATURE = "structured_answer"


@pytest.fixture
def validator():
    return SchemaValidator(load_policies().get("schema_mismatch"))


def test_valid_output_has_no_findings(validator):
    output = '{"answer": "The weather may improve tomorrow.", "confidence": 0.72}'
    assert validator.validate(FEATURE, output) == []


def test_integer_confidence_is_accepted(validator):
    assert validator.validate(FEATURE, '{"answer": "Yes.", "confidence": 1}') == []


def test_missing_field_is_reported(validator):
    output = '{"answer": "The weather may improve tomorrow."}'
    findings = validator.validate(FEATURE, output)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.policy == "schema_mismatch"
    assert finding.severity == Severity.MEDIUM
    assert finding.action == Decision.REWRITE
    assert "confidence" in finding.reason


def test_invalid_json_is_reported(validator):
    findings = validator.validate(FEATURE, "The answer is yes, confidence 0.9")
    assert len(findings) == 1
    assert "not valid JSON" in findings[0].reason


def test_empty_output_is_reported(validator):
    assert len(validator.validate(FEATURE, "")) == 1


def test_wrong_type_is_reported(validator):
    findings = validator.validate(FEATURE, '{"answer": "Yes.", "confidence": "high"}')
    assert len(findings) == 1
    assert "confidence" in findings[0].reason


def test_string_number_is_not_coerced(validator):
    assert len(validator.validate(FEATURE, '{"answer": "Yes.", "confidence": "0.9"}')) == 1


def test_out_of_range_confidence_is_reported(validator):
    findings = validator.validate(FEATURE, '{"answer": "Yes.", "confidence": 1.5}')
    assert len(findings) == 1
    assert "confidence" in findings[0].reason


def test_extra_field_is_reported(validator):
    output = '{"answer": "Yes.", "confidence": 0.9, "debug": true}'
    findings = validator.validate(FEATURE, output)
    assert len(findings) == 1
    assert "debug" in findings[0].reason


def test_non_object_json_is_reported(validator):
    assert len(validator.validate(FEATURE, "[1, 2, 3]")) == 1


def test_multiple_errors_give_one_finding(validator):
    findings = validator.validate(FEATURE, '{"confidence": 5}')
    assert len(findings) == 1
    assert "answer" in findings[0].reason and "confidence" in findings[0].reason


def test_feature_without_schema_is_skipped(validator):
    assert validator.validate("free_text_feature", "anything at all") == []


def test_evidence_is_a_span_of_the_output_and_truncated(validator):
    output = "not json " + "x" * 500
    finding = validator.validate(FEATURE, output)[0]
    assert finding.evidence in output
    assert len(finding.evidence) <= 200


def test_policy_changes_flow_through(validator):
    # Severity/action come from the policy object, not hard-coded.
    policy = load_policies().get("schema_mismatch").model_copy(
        update={"severity": Severity.HIGH, "action": Decision.BLOCK}
    )
    finding = SchemaValidator(policy).validate(FEATURE, "nope")[0]
    assert finding.severity == Severity.HIGH
    assert finding.action == Decision.BLOCK
