from ragagent.planner import classify, plan


def test_plan_single_question_keeps_anchor_and_keyword():
    subs = plan("What is the return window?")
    assert subs[0].text == "What is the return window?"
    assert any(sq.kind == "keyword" for sq in subs)


def test_plan_splits_multi_part_question():
    subs = plan("What is the return window and how much does express "
                "shipping cost?")
    texts = [sq.text for sq in subs]
    assert "What is the return window?" in texts
    assert "How much does express shipping cost?" in texts


def test_plan_does_not_split_plain_and():
    subs = plan("If I buy the trail jacket at $129.99 and the hiking boots "
                "at $159.99, what is the total before tax?")
    assert len([sq for sq in subs if sq.origin == "planner"]) <= 2
    assert subs[0].text.startswith("If I buy")


def test_plan_empty_question():
    assert plan("   ") == []


def test_classify_kinds():
    assert classify("What is the total of 12 and 30?") == "arithmetic"
    assert classify("What day of the week is 2026-11-17?") == "date"
    assert classify("What is a warranty?") == "definition"
    assert classify("How much does shipping cost?") == "factual"


def test_plan_dedupes_and_caps():
    subs = plan("What is the return window?", max_subqueries=1)
    assert len(subs) == 1
