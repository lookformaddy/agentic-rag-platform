"""Golden-set eval harness for the agentic RAG platform.

Each golden case carries the expected answer keywords, the expected
source documents, and whether the correct behaviour is abstention.
Four metrics per case:

- answer accuracy — fraction of expected keywords present in the answer
  (0 when the answer abstains and abstention was not expected);
- citation precision — fraction of cited sources that are expected
  (1.0 when the answer correctly abstains with no citations);
- abstention correctness — predicted abstention matches expectation;
- retrieval recall@k — fraction of expected sources in the top-k hits.
"""
from __future__ import annotations

from ragagent.models import (AnswerStatus, EvalCase, EvalCaseResult,
                             EvalReport)


def _keyword_accuracy(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    text = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def _citation_precision(cited_sources: list[str],
                        expected_sources: list[str]) -> float:
    if not cited_sources:
        return 1.0
    expected = set(expected_sources)
    hits = sum(1 for s in cited_sources if s in expected)
    return hits / len(cited_sources)


def evaluate(engine, cases: list[EvalCase], top_k: int = 5) -> EvalReport:
    """Run the golden set through ``engine`` and aggregate the metrics."""
    results: list[EvalCaseResult] = []
    for case in cases:
        answer = engine.ask(case.question, tenant=case.tenant)
        cited = [ev.source.split("#")[0] for ev in answer.citations]
        cited = [c for c in cited if not c.startswith("tool:")]
        recalled = engine.store.recall_at_k(
            case.question, case.expected_sources, top_k=top_k,
            vector_weight=engine.policy.vector_weight)
        if answer.status == AnswerStatus.ABSTAINED:
            accuracy = 1.0 if case.expect_abstain else 0.0
        else:
            accuracy = _keyword_accuracy(answer.text, case.expected_keywords)
        results.append(EvalCaseResult(
            case_id=case.case_id,
            answer_accuracy=accuracy,
            citation_precision=_citation_precision(cited, case.expected_sources)
            if answer.status != AnswerStatus.ABSTAINED else 1.0,
            abstention_correct=(answer.status == AnswerStatus.ABSTAINED)
            == case.expect_abstain,
            retrieval_recall_at_k=recalled,
            abstained=answer.status == AnswerStatus.ABSTAINED,
            top_k_sources=sorted({h.chunk.doc_id for h in
                                  engine.last_hits}) if hasattr(engine, "last_hits") else [],
        ))
    n = len(results) or 1
    return EvalReport(
        policy_version=engine.policy.version,
        n_cases=len(results),
        mean_answer_accuracy=sum(r.answer_accuracy for r in results) / n,
        mean_citation_precision=sum(r.citation_precision for r in results) / n,
        abstention_correctness=sum(1 for r in results if r.abstention_correct) / n,
        mean_recall_at_k=sum(r.retrieval_recall_at_k for r in results) / n,
        cases=results,
    )
