from ragagent.eval import evaluate
from ragagent.models import AnswerStatus


def test_eval_golden_set_quality(engine, cases):
    report = evaluate(engine, cases)
    assert report.n_cases == 10
    assert report.mean_answer_accuracy >= 0.9
    assert report.mean_citation_precision >= 0.8
    assert report.abstention_correctness == 1.0
    assert report.mean_recall_at_k >= 0.8


def test_eval_abstain_case_is_correct(engine, cases):
    report = evaluate(engine, cases)
    abstain_case = next(c for c in report.cases if c.case_id == "abstain-revenue")
    assert abstain_case.abstention_correct
    assert abstain_case.abstained


def test_eval_arithmetic_case_uses_tool(engine, cases):
    report = evaluate(engine, cases)
    case = next(c for c in report.cases if c.case_id == "arithmetic-total")
    assert case.answer_accuracy == 1.0


def test_eval_date_case_answers_weekday(engine, cases):
    report = evaluate(engine, cases)
    case = next(c for c in report.cases if c.case_id == "date-weekday")
    assert case.answer_accuracy == 1.0
