"""Optional FastAPI service surface.

Requires fastapi + httpx (installed via requirements.txt). The test suite
auto-skips this module when FastAPI is not installed; it runs in CI.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragagent.demo import build_engine
from ragagent.demo_data import golden_set
from ragagent.engine import QueryEngine
from ragagent.eval import evaluate
from ragagent.models import AnswerStatus, Document


class AskRequest(BaseModel):
    question: str
    tenant: str = "default"


class DocumentIn(BaseModel):
    doc_id: str
    title: str
    text: str
    source: str = "corpus"
    tags: list[str] = Field(default_factory=list)


class PolicyIn(BaseModel):
    version: str
    name: str
    vector_weight: float = 0.5
    top_k: int = 5
    sufficiency_threshold: float = 0.30
    abstain_threshold: float = 0.25
    min_evidence_score: float = 0.20
    max_rounds: int = 3
    notes: str = ""


def create_app(engine: QueryEngine | None = None) -> FastAPI:
    engine = engine or build_engine()
    app = FastAPI(title="Agentic RAG Platform",
                  description="Agentic retrieval-augmented generation with "
                              "self-critique, tools, budgets and eval")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok",
                "chunks": len(engine.store),
                "active_policy": engine.policy.version,
                "audit_entries": len(engine.audit.entries)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        answer = engine.ask(req.question, tenant=req.tenant)
        if answer.status == AnswerStatus.BLOCKED:
            raise HTTPException(status_code=429, detail=answer.abstain_reason)
        return {"request_id": answer.request_id,
                "question": answer.question,
                "status": answer.status.value,
                "text": answer.text,
                "citations": [{"source": c.source, "text": c.text,
                               "score": round(c.score, 3)}
                              for c in answer.citations],
                "sub_queries": [sq.text for sq in answer.sub_queries],
                "rounds": answer.rounds,
                "tool_calls": [{"tool": t.tool, "ok": t.ok,
                                "result": t.result, "error": t.error}
                               for t in answer.tool_calls],
                "sufficiency": round(answer.sufficiency, 3),
                "tokens_in": answer.tokens_in,
                "tokens_out": answer.tokens_out,
                "cost_usd": answer.cost_usd,
                "latency_ms": answer.latency_ms,
                "abstain_reason": answer.abstain_reason,
                "policy_version": answer.policy_version}

    @app.post("/ingest")
    def ingest(doc: DocumentIn) -> dict[str, Any]:
        n = engine.store.add_document(Document(
            doc_id=doc.doc_id, title=doc.title, text=doc.text,
            source=doc.source, tags=tuple(doc.tags)))
        return {"doc_id": doc.doc_id, "chunks": n,
                "total_chunks": len(engine.store)}

    @app.get("/policies")
    def list_policies() -> dict[str, Any]:
        return {"policies": [
            {"version": p.version, "name": p.name,
             "vector_weight": p.vector_weight, "top_k": p.top_k,
             "sufficiency_threshold": p.sufficiency_threshold,
             "abstain_threshold": p.abstain_threshold,
             "status": p.status.value, "notes": p.notes}
            for p in engine.policies.list()]}

    @app.post("/policies")
    def register_policy(body: PolicyIn) -> dict[str, Any]:
        from ragagent.models import PolicyStatus, RetrievalPolicy
        try:
            engine.policies.register(RetrievalPolicy(
                version=body.version, name=body.name,
                vector_weight=body.vector_weight, top_k=body.top_k,
                sufficiency_threshold=body.sufficiency_threshold,
                abstain_threshold=body.abstain_threshold,
                min_evidence_score=body.min_evidence_score,
                max_rounds=body.max_rounds,
                status=PolicyStatus.DRAFT, notes=body.notes))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"version": body.version, "status": "draft"}

    @app.post("/policies/{version}/activate")
    def activate_policy(version: str) -> dict[str, Any]:
        try:
            policy = engine.policies.activate(version)
            engine.set_policy(version)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"version": policy.version, "status": policy.status.value}

    @app.post("/eval")
    def run_eval() -> dict[str, Any]:
        report = evaluate(engine, golden_set())
        return {"policy_version": report.policy_version,
                "n_cases": report.n_cases,
                "mean_answer_accuracy": round(report.mean_answer_accuracy, 3),
                "mean_citation_precision": round(report.mean_citation_precision, 3),
                "abstention_correctness": round(report.abstention_correctness, 3),
                "mean_recall_at_k": round(report.mean_recall_at_k, 3),
                "cases": [{"case_id": c.case_id,
                           "answer_accuracy": round(c.answer_accuracy, 3),
                           "citation_precision": round(c.citation_precision, 3),
                           "abstention_correct": c.abstention_correct,
                           "recall_at_k": round(c.retrieval_recall_at_k, 3)}
                          for c in report.cases]}

    @app.get("/ledger/{tenant}")
    def tenant_ledger(tenant: str) -> dict[str, Any]:
        usage = engine.ledger.tenant_usage(tenant)
        return {"tenant": tenant, "cost_usd": round(usage.cost_usd, 6),
                "requests": usage.requests, "tokens": usage.tokens,
                "cost_cap_usd": engine.ledger.cost_cap_usd,
                "request_cap": engine.ledger.request_cap}

    @app.get("/audit/verify")
    def audit_verify() -> dict[str, Any]:
        ok, message = engine.audit.verify()
        return {"valid": ok, "message": message,
                "entries": len(engine.audit.entries)}

    return app


app = create_app()
