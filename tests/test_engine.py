from ragagent.models import AnswerStatus, RetrievalPolicy


def test_ask_simple_question_answered_with_citation(engine):
    answer = engine.ask("What is the return window?")
    assert answer.status == AnswerStatus.ANSWERED
    assert "30" in answer.text
    assert any(c.source.startswith("returns-policy") for c in answer.citations)
    assert answer.rounds >= 1
    assert answer.cost_usd > 0
    assert answer.policy_version == "v1.0.0"


def test_ask_multi_part_decomposes(engine):
    answer = engine.ask("What is the return window and how much does "
                        "express shipping cost?")
    assert answer.status == AnswerStatus.ANSWERED
    assert "30" in answer.text and "12.99" in answer.text
    assert len(answer.sub_queries) >= 2


def test_ask_arithmetic_uses_calculator(engine):
    answer = engine.ask("If I buy the trail jacket at $129.99 and the "
                        "hiking boots at $159.99, what is the total "
                        "before tax?")
    assert answer.status == AnswerStatus.ANSWERED
    assert "289.98" in answer.text
    assert any(t.tool == "calculator" and t.ok for t in answer.tool_calls)


def test_ask_date_question_uses_date_tool(engine):
    answer = engine.ask("What day of the week is 2026-11-17?")
    assert answer.status == AnswerStatus.ANSWERED
    assert "Tuesday" in answer.text
    assert any(t.tool == "date_lookup" and t.ok for t in answer.tool_calls)


def test_ask_definition_question_uses_glossary(engine):
    answer = engine.ask("What is a warranty?")
    assert answer.status == AnswerStatus.ANSWERED
    assert "1-year" in answer.text


def test_ask_abstains_on_out_of_corpus_question(engine):
    answer = engine.ask("What is the quarterly revenue of Northwind Outfitters?")
    assert answer.status == AnswerStatus.ABSTAINED
    assert answer.text == ""
    assert answer.abstain_reason != ""


def test_ask_records_ledger_and_audit(engine):
    before = len(engine.ledger.entries)
    answer = engine.ask("What is the return window?", tenant="team-a")
    assert len(engine.ledger.entries) == before + 1
    usage = engine.ledger.tenant_usage("team-a")
    assert usage.requests >= 1 and usage.cost_usd > 0
    events = [e.event for e in engine.audit.events()]
    assert "retrieval_round" in events and "answer" in events
    ok, _ = engine.audit.verify()
    assert ok


def test_ask_blocked_when_budget_exhausted(engine):
    engine.ledger.cost_cap_usd = 0.00001
    answer = engine.ask("What is the return window?")
    assert answer.status == AnswerStatus.BLOCKED
    assert "cost cap" in answer.abstain_reason


def test_set_policy_switches_retrieval(engine):
    engine.policies.register(
        RetrievalPolicy(version="v9.9.9", name="wide", top_k=8))
    engine.set_policy("v9.9.9")
    assert engine.policy.version == "v9.9.9"
    # Registry active policy is untouched by set_policy (only promotion
    # changes it).
    assert engine.policies.active().version == "v1.0.0"
