import pytest

from ragagent.ledger import BudgetExceeded, Ledger, LedgerEntry, estimate_tokens


def _ledger(**kw):
    kw.setdefault("cost_cap_usd", 1.0)
    kw.setdefault("request_cap", 10)
    kw.setdefault("token_cap", 10_000)
    return Ledger(**kw)


def test_estimate_tokens_scales_with_words():
    assert estimate_tokens("one two three four") == 5
    assert estimate_tokens("") == 1


def test_check_passes_within_budget():
    _ledger().check("team-a", est_cost_usd=0.5, est_tokens=100)


def test_check_blocks_cost_cap():
    ledger = _ledger(cost_cap_usd=0.01)
    ledger.record(LedgerEntry(request_id="r1", tenant="team-a", tokens_in=10,
                              tokens_out=5, cost_usd=0.009, latency_ms=1,
                              rounds=1, tool_calls=0, status="answered"))
    with pytest.raises(BudgetExceeded, match="cost cap"):
        ledger.check("team-a", est_cost_usd=0.005)


def test_check_blocks_request_cap():
    ledger = _ledger(request_cap=1)
    ledger.record(LedgerEntry(request_id="r1", tenant="team-a", tokens_in=1,
                              tokens_out=1, cost_usd=0.0, latency_ms=1,
                              rounds=1, tool_calls=0, status="answered"))
    with pytest.raises(BudgetExceeded, match="request cap"):
        ledger.check("team-a")


def test_record_accumulates_usage():
    ledger = _ledger()
    ledger.record(LedgerEntry(request_id="r1", tenant="team-a", tokens_in=100,
                              tokens_out=50, cost_usd=0.01, latency_ms=10,
                              rounds=2, tool_calls=1, status="answered"))
    usage = ledger.tenant_usage("team-a")
    assert usage.requests == 1 and usage.tokens == 150
    assert usage.cost_usd == 0.01
    assert ledger.total_cost() == 0.01
