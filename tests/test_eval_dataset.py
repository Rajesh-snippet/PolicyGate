import pytest

from app.pipeline import DeterministicChecks
from app.policies import load_policies
from evals.run_eval import detected_policies, evaluate, load_dataset


@pytest.fixture(scope="module")
def registry():
    return load_policies()


@pytest.fixture(scope="module")
def examples():
    return load_dataset()


def test_dataset_is_well_formed(examples, registry):
    assert len(examples) >= 20
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))
    known = {p.name for p in registry.all()}
    for example in examples:
        assert set(example.expected_policies) <= known
    assert {e.layer for e in examples} == {"deterministic", "judge"}


def test_deterministic_examples_match_their_labels(examples, registry):
    checks = DeterministicChecks.from_registry(registry)
    for example in examples:
        if example.layer == "deterministic":
            found = detected_policies(checks, example)
            assert found == set(example.expected_policies), example.id


def test_metrics_are_perfect_on_the_starter_set(examples, registry):
    checks = DeterministicChecks.from_registry(registry)
    metrics = evaluate(examples, checks, [p.name for p in registry.all()])
    for name, m in metrics.items():
        assert m.fp == 0 and m.fn == 0, name
    assert metrics["pii_leakage"].recall == 1.0
    assert metrics["pii_leakage"].false_positive_rate == 0.0
