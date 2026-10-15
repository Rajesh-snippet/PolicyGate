import pytest

from app.models import Decision, Severity
from app.policies import load_policies
from app.validators.pii import PIIDetector

FEATURE = "health_assistant"  # no PII overrides


@pytest.fixture
def policy():
    return load_policies().get("pii_leakage")


@pytest.fixture
def detector(policy):
    return PIIDetector(policy)


def evidence(findings):
    return [f.evidence for f in findings]


def reasons(findings):
    return {f.reason for f in findings}


def test_email_finding_shape(detector):
    output = "Reach me at rajesh@gmail.com for details."
    findings = detector.detect(FEATURE, output)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.policy == "pii_leakage"
    assert finding.severity == Severity.HIGH
    assert finding.action == Decision.REWRITE
    assert finding.evidence == "rajesh@gmail.com"
    assert finding.evidence in output


def test_clean_text_has_no_findings(detector):
    text = "In 2024 we sold 1234567 units. Python 3.11 was released on 2022-10-24."
    assert detector.detect(FEATURE, text) == []


@pytest.mark.parametrize(
    "phone",
    ["+91 98765 43210", "9876543210", "(415) 555-0142", "415-555-0142", "555-0142", "+1 415 555 0142"],
)
def test_phone_numbers_are_detected(detector, phone):
    findings = detector.detect(FEATURE, f"Call {phone} today.")
    assert evidence(findings) == [phone]
    assert reasons(findings) == {"phone detected"}


def test_valid_card_is_detected_once_and_not_as_phone(detector):
    findings = detector.detect(FEATURE, "Card: 4111 1111 1111 1111.")
    assert evidence(findings) == ["4111 1111 1111 1111"]
    assert reasons(findings) == {"credit_card detected"}


def test_card_like_number_failing_luhn_is_ignored(detector):
    assert detector.detect(FEATURE, "Order 4111 1111 1111 1112 shipped.") == []


def test_api_keys_are_detected(detector):
    openai_style = "sk-" + "a1B2c3D4" * 4
    aws = "AKIAIOSFODNN7EXAMPLE"
    github = "ghp_" + "a1B2c3" * 6
    findings = detector.detect(FEATURE, f"keys: {openai_style} {aws} {github}")
    assert set(evidence(findings)) == {openai_style, aws, github}
    assert reasons(findings) == {"api_key detected"}


def test_secret_assignment_reports_only_the_value(detector):
    value = "abcd1234" * 3
    findings = detector.detect(FEATURE, f'config: api_key = "{value}"')
    assert evidence(findings) == [value]


def test_short_password_value_is_not_flagged(detector):
    assert detector.detect(FEATURE, "password: hunter2") == []


def test_ip_addresses(detector):
    findings = detector.detect(FEATURE, "Server at 192.168.1.10. Not 999.1.1.1 though.")
    assert evidence(findings) == ["192.168.1.10"]


def test_duplicates_are_reported_once(detector):
    findings = detector.detect(FEATURE, "a@b.com and again a@b.com")
    assert len(findings) == 1


def test_multiple_types_in_one_output(detector):
    findings = detector.detect(FEATURE, "john@acme.com at 555-0142")
    assert reasons(findings) == {"email detected", "phone detected"}


def test_feature_can_allow_email_but_not_phone(detector):
    output = "Account owner: john@acme.com, phone 415-555-0142"
    findings = detector.detect("support_assistant", output)
    assert reasons(findings) == {"phone detected"}
    # same output for a feature without overrides flags both
    assert reasons(detector.detect(FEATURE, output)) == {"email detected", "phone detected"}


def test_secrets_and_cards_can_never_be_allowed(policy):
    detector = PIIDetector(policy, feature_allowed={"f": {"api_key", "credit_card", "email"}})
    output = "a@b.com 4111 1111 1111 1111 AKIAIOSFODNN7EXAMPLE"
    assert reasons(detector.detect("f", output)) == {
        "credit_card detected",
        "api_key detected",
    }


def test_overrides_ignored_when_policy_disallows_them(policy):
    strict = policy.model_copy(update={"per_feature_override": False})
    findings = PIIDetector(strict).detect("support_assistant", "john@acme.com")
    assert reasons(findings) == {"email detected"}


def test_unknown_detector_in_policy_fails_fast(policy):
    bad = policy.model_copy(update={"detectors": ["email", "passport"]})
    with pytest.raises(ValueError):
        PIIDetector(bad)


def test_only_detectors_named_in_policy_run(policy):
    narrow = policy.model_copy(update={"detectors": ["email"]})
    findings = PIIDetector(narrow).detect(FEATURE, "a@b.com 415-555-0142")
    assert reasons(findings) == {"email detected"}