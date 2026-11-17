import pytest

from ragagent.models import EvalReport, PolicyStatus, RetrievalPolicy
from ragagent.policies import (PolicyRegistry, compare_reports,
                               gated_promote)


def _policy(version, **kw):
    kw.setdefault("status", PolicyStatus.DRAFT)
    return RetrievalPolicy(version=version, name=version, **kw)


def _report(recall=0.9, precision=0.9, abstain=1.0, accuracy=0.9):
    return EvalReport(policy_version="vX", n_cases=10,
                      mean_answer_accuracy=accuracy,
                      mean_citation_precision=precision,
                      abstention_correctness=abstain,
                      mean_recall_at_k=recall)


def test_registry_keeps_one_active():
    registry = PolicyRegistry(_policy("v1.0.0"))
    registry.register(_policy("v1.1.0"))
    registry.activate("v1.1.0")
    assert registry.active().version == "v1.1.0"
    assert registry.get("v1.0.0").status == PolicyStatus.DEPRECATED


def test_registry_rejects_duplicate_version():
    registry = PolicyRegistry(_policy("v1.0.0"))
    with pytest.raises(ValueError, match="already exists"):
        registry.register(_policy("v1.0.0"))


def test_gated_promote_passes_and_activates():
    registry = PolicyRegistry(_policy("v1.0.0"))
    registry.register(_policy("v1.1.0"))
    result = gated_promote(registry, "v1.1.0", _report(),
                           baseline_report=_report())
    assert result.promoted
    assert registry.active().version == "v1.1.0"


def test_gated_promote_blocks_on_floor():
    registry = PolicyRegistry(_policy("v1.0.0"))
    registry.register(_policy("v1.1.0"))
    result = gated_promote(registry, "v1.1.0", _report(recall=0.5),
                           baseline_report=_report())
    assert not result.promoted
    assert any("recall@k" in r for r in result.reasons)
    assert registry.active().version == "v1.0.0"


def test_gated_promote_blocks_on_regression():
    registry = PolicyRegistry(_policy("v1.0.0"))
    registry.register(_policy("v1.1.0"))
    result = gated_promote(registry, "v1.1.0", _report(precision=0.8),
                           baseline_report=_report(precision=0.95))
    assert not result.promoted
    assert any("citation_precision" in r for r in result.reasons)


def test_compare_reports_deltas():
    deltas = compare_reports(_report(recall=0.9), _report(recall=0.8))
    assert deltas["recall_at_k"] == pytest.approx(0.1)
