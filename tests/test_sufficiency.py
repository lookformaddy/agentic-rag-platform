from ragagent.models import SubQuery
from ragagent.planner import plan
from ragagent.sufficiency import coverage_for, follow_ups, sufficiency


def test_coverage_high_for_grounded_question(store):
    sq = SubQuery(text="What is the return window?")
    assert coverage_for(sq, store) >= 0.35


def test_coverage_low_for_out_of_corpus_question(store):
    sq = SubQuery(text="What is the quarterly revenue of Northwind Outfitters?")
    assert coverage_for(sq, store) < 0.32


def test_sufficiency_is_weakest_link(store):
    subs = plan("What is the return window and how much does express "
                "shipping cost?")
    overall, per = sufficiency(subs, store)
    assert overall == min(per.values())
    assert overall >= 0.35


def test_sufficiency_empty_subqueries(store):
    overall, per = sufficiency([], store)
    assert overall == 0.0 and per == {}


def test_follow_ups_include_keyword_and_synonym_forms():
    uncovered = [SubQuery(text="What is the return window?", kind="factual")]
    ups = follow_ups(uncovered)
    texts = [sq.text for sq in ups]
    assert "return window" in texts
    assert any("refund" in t for t in texts)
    assert all(sq.origin == "followup" for sq in ups)
