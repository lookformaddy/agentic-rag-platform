from ragagent.generator import generate
from ragagent.models import AnswerStatus, ToolCallResult


def test_generate_answers_with_citations(store):
    hits = store.retrieve("What is the return window?", top_k=5)
    status, text, citations, reason = generate(
        "What is the return window?", 0.6, hits, [],
        abstain_threshold=0.32, min_evidence_score=0.10)
    assert status == AnswerStatus.ANSWERED
    assert "30" in text
    assert citations and all("[returns-policy#" in c.source or
                             c.source.startswith("returns-policy")
                             for c in citations)
    assert reason == ""


def test_generate_abstains_below_threshold(store):
    hits = store.retrieve("What is the quarterly revenue?", top_k=5)
    status, text, citations, reason = generate(
        "What is the quarterly revenue?", 0.2, hits, [],
        abstain_threshold=0.32, min_evidence_score=0.10)
    assert status == AnswerStatus.ABSTAINED
    assert text == "" and citations == [] and reason != ""


def test_generate_uses_tool_evidence_when_retrieval_is_thin(store):
    hits = store.retrieve("What is the quarterly revenue?", top_k=5)
    tool = ToolCallResult(tool="calculator", args={"expression": "2 + 2"},
                          ok=True, result="2 + 2 = 4")
    status, text, citations, _ = generate(
        "What is 2 + 2?", 0.1, hits, [tool],
        abstain_threshold=0.32, min_evidence_score=0.10)
    assert status == AnswerStatus.ANSWERED
    assert "2 + 2 = 4" in text
    assert any(c.source == "tool:calculator" for c in citations)


def test_generate_abstains_when_no_evidence_clears_floor(store):
    hits = store.retrieve("What is the quarterly revenue?", top_k=5)
    status, _, _, reason = generate(
        "What is the quarterly revenue?", 0.9, hits, [],
        abstain_threshold=0.32, min_evidence_score=0.99)
    assert status == AnswerStatus.ABSTAINED
    assert "minimum evidence score" in reason
