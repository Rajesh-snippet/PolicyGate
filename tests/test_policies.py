import pytest

from app.models import Decision, Severity
from app.policies import PolicyLoadError, Strategy, load_policies

VALID = """
name: custom_rule
version: 1.0.0
severity: low
strategy: deterministic
action: rewrite
patterns:
  - "foo"
"""


def test_loads_all_default_policies():
    registry = load_policies()
    assert {p.name for p in registry.all()} == {
        "pii_leakage",
        "unsafe_instruction",
        "medical_overconfidence",
        "unsupported_claims",
        "brand_tone",
        "schema_mismatch",
    }
    assert len(registry.version) == 12


def test_policy_fields_are_parsed():
    policy = load_policies().get("medical_overconfidence")
    assert policy.severity == Severity.HIGH
    assert policy.action == Decision.REWRITE  # "rewrite" in YAML -> REWRITE
    assert policy.fallback_action == Decision.BLOCK
    assert "will definitely cure" in policy.patterns
    assert policy.examples.bad


def test_by_strategy():
    registry = load_policies()
    judged = {p.name for p in registry.by_strategy(Strategy.LLM_JUDGE, Strategy.HYBRID)}
    assert judged == {"brand_tone", "unsupported_claims"}


def test_policies_change_without_code_changes(tmp_path):
    (tmp_path / "custom.yaml").write_text(VALID)
    registry = load_policies(tmp_path)
    assert registry.get("custom_rule").patterns == ["foo"]


def test_version_changes_when_a_policy_changes(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text(VALID)
    before = load_policies(tmp_path).version
    path.write_text(VALID.replace('"foo"', '"bar"'))
    assert load_policies(tmp_path).version != before


def test_invalid_regex_is_rejected(tmp_path):
    (tmp_path / "bad.yaml").write_text(VALID.replace('"foo"', '"(unclosed"'))
    with pytest.raises(PolicyLoadError):
        load_policies(tmp_path)


def test_deterministic_policy_needs_a_rule(tmp_path):
    (tmp_path / "bad.yaml").write_text(VALID.replace('patterns:\n  - "foo"\n', ""))
    with pytest.raises(PolicyLoadError):
        load_policies(tmp_path)


def test_duplicate_names_are_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text(VALID)
    (tmp_path / "b.yaml").write_text(VALID)
    with pytest.raises(PolicyLoadError):
        load_policies(tmp_path)


def test_empty_directory_is_rejected(tmp_path):
    with pytest.raises(PolicyLoadError):
        load_policies(tmp_path)
