"""Demo scenarios and engine factory for the Agentic RAG Platform."""
from __future__ import annotations

from ragagent.audit import AuditLog
from ragagent.config import Settings
from ragagent.demo_data import GLOSSARY, demo_documents, golden_set
from ragagent.engine import QueryEngine
from ragagent.eval import evaluate
from ragagent.ledger import Ledger
from ragagent.models import AnswerStatus, PolicyStatus, RetrievalPolicy
from ragagent.policies import PolicyRegistry, gated_promote
from ragagent.store import DocumentStore
from ragagent.tools import ToolRegistry


def default_policy(settings: Settings) -> RetrievalPolicy:
    return RetrievalPolicy(
        version="v1.0.0", name="balanced",
        vector_weight=settings.default_vector_weight,
        top_k=settings.default_top_k,
        sufficiency_threshold=settings.default_sufficiency_threshold,
        abstain_threshold=settings.default_abstain_threshold,
        min_evidence_score=settings.default_min_evidence_score,
        max_rounds=settings.default_max_rounds,
        status=PolicyStatus.ACTIVE,
        notes="Balanced hybrid retrieval (0.5 vector / 0.5 keyword).")


def build_engine(settings: Settings | None = None) -> QueryEngine:
    """Build a fully-wired engine over the synthetic demo corpus."""
    settings = settings or Settings()
    store = DocumentStore(dim=settings.embedding_dim)
    store.add_documents(demo_documents())
    tools = ToolRegistry(reference_date=settings.reference_date,
                         glossary=GLOSSARY)
    ledger = Ledger(cost_cap_usd=settings.tenant_cost_cap_usd,
                    request_cap=settings.tenant_request_cap,
                    token_cap=settings.tenant_token_cap)
    audit = AuditLog()
    policies = PolicyRegistry(default_policy(settings))
    return QueryEngine(settings=settings, store=store, tools=tools,
                       ledger=ledger, audit=audit, policies=policies)


def run_demo() -> dict:
    """Run the six demo scenarios end to end. Returns a results dict."""
    engine = build_engine()
    results: dict = {"scenarios": []}

    def scenario(name: str, ok: bool, detail: str) -> None:
        results["scenarios"].append({"name": name, "ok": ok, "detail": detail})
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")

    # 1. Simple factual question, answered with citations.
    a1 = engine.ask("What is the return window?")
    scenario("simple factual answer",
             a1.status == AnswerStatus.ANSWERED and "30" in a1.text
             and any(c.source.startswith("returns-policy") for c in a1.citations),
             f"status={a1.status.value} sufficiency={a1.sufficiency:.2f} "
             f"rounds={a1.rounds} citations={[c.source for c in a1.citations]}")

    # 2. Multi-part question: planner decomposes, both parts grounded.
    a2 = engine.ask("What is the return window and how much does "
                    "express shipping cost?")
    scenario("multi-part decomposition",
             a2.status == AnswerStatus.ANSWERED and "30" in a2.text
             and "12.99" in a2.text and len(a2.sub_queries) >= 2,
             f"sub_queries={[sq.text for sq in a2.sub_queries]} "
             f"rounds={a2.rounds} sufficiency={a2.sufficiency:.2f}")

    # 3. Arithmetic gap: retrieval states the prices, the calculator tool
    #    supplies the total the text cannot.
    a3 = engine.ask("If I buy the trail jacket at $129.99 and the hiking "
                    "boots at $159.99, what is the total before tax?")
    scenario("tool-backed arithmetic",
             a3.status == AnswerStatus.ANSWERED and "289.98" in a3.text
             and any(t.tool == "calculator" and t.ok for t in a3.tool_calls),
             f"tools={[(t.tool, t.result) for t in a3.tool_calls]} "
             f"rounds={a3.rounds}")

    # 4. Out-of-corpus question: abstain, never guess.
    a4 = engine.ask("What is the quarterly revenue of Northwind Outfitters?")
    scenario("abstention on thin evidence",
             a4.status == AnswerStatus.ABSTAINED and a4.abstain_reason != "",
             f"status={a4.status.value} sufficiency={a4.sufficiency:.2f} "
             f"reason={a4.abstain_reason}")

    # 5. Fail-closed tenant budget: a near-zero cap blocks the request.
    capped_settings = Settings()
    capped = build_engine(capped_settings)
    capped.ledger.cost_cap_usd = 0.00001
    a5 = capped.ask("What is the return window?")
    scenario("fail-closed tenant budget",
             a5.status == AnswerStatus.BLOCKED,
             f"status={a5.status.value} reason={a5.abstain_reason}")

    # 6. Golden-set eval + gated A/B promotion of a recall-heavy policy.
    baseline = evaluate(engine, golden_set())
    candidate = RetrievalPolicy(
        version="v1.1.0", name="recall-heavy", vector_weight=0.5, top_k=8,
        sufficiency_threshold=engine.policy.sufficiency_threshold,
        abstain_threshold=engine.policy.abstain_threshold,
        min_evidence_score=engine.policy.min_evidence_score,
        max_rounds=engine.policy.max_rounds,
        notes="Same weights, wider top-k for higher recall.")
    engine.policies.register(candidate)
    engine.set_policy("v1.1.0")
    candidate_report = evaluate(engine, golden_set())
    engine.set_policy("v1.0.0")
    promotion = gated_promote(engine.policies, "v1.1.0", candidate_report,
                              baseline_report=baseline)
    results["baseline_eval"] = {
        "answer_accuracy": round(baseline.mean_answer_accuracy, 3),
        "citation_precision": round(baseline.mean_citation_precision, 3),
        "abstention_correctness": round(baseline.abstention_correctness, 3),
        "recall_at_k": round(baseline.mean_recall_at_k, 3)}
    results["candidate_eval"] = {
        "answer_accuracy": round(candidate_report.mean_answer_accuracy, 3),
        "citation_precision": round(candidate_report.mean_citation_precision, 3),
        "abstention_correctness": round(candidate_report.abstention_correctness, 3),
        "recall_at_k": round(candidate_report.mean_recall_at_k, 3)}
    scenario("golden-set eval + gated promotion",
             baseline.mean_recall_at_k >= 0.70 and promotion.promoted,
             f"baseline recall@k={baseline.mean_recall_at_k:.3f} "
             f"candidate recall@k={candidate_report.mean_recall_at_k:.3f} "
             f"promoted={promotion.promoted} reasons={promotion.reasons} "
             f"active={engine.policies.active().version if engine.policies.active() else None}")

    # 7. Audit chain verifies end to end.
    ok, message = engine.audit.verify()
    scenario("hash-chained audit verifies", ok, message)

    ledger = engine.ledger
    results["ledger"] = {
        "requests": len(ledger.entries),
        "total_cost_usd": round(ledger.total_cost(), 6),
        "total_tokens": sum(e.tokens_in + e.tokens_out for e in ledger.entries)}
    results["all_ok"] = all(s["ok"] for s in results["scenarios"])
    print(f"\nDemo: {sum(1 for s in results['scenarios'] if s['ok'])}/"
          f"{len(results['scenarios'])} scenarios passed; "
          f"ledger: {results['ledger']['requests']} requests, "
          f"${results['ledger']['total_cost_usd']:.6f}, "
          f"{results['ledger']['total_tokens']} tokens.")
    return results
