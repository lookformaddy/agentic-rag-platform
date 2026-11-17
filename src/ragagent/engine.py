"""Query engine: the agentic loop.

One request runs:

1. **Budget check (fail-closed)** — the tenant's cost/request/token caps
   are checked before anything executes.
2. **Plan** — the question is decomposed into sub-queries.
3. **Retrieve → critique → repeat** — each round retrieves against every
   open sub-query, scores sufficiency (weakest sub-query), and, while
   evidence is thin and rounds remain, generates follow-up reformulations
   and retrieves again.
4. **Tools** — if retrieval is still insufficient, schema-validated
   scripted tools are called for the uncovered sub-queries (arithmetic,
   date, definition).
5. **Generate or abstain** — an extractive, citation-attached answer, or
   an explicit abstention when evidence is thin.
6. **Account + audit** — tokens, cost and latency are recorded in the
   ledger; every step is appended to the hash-chained audit log.
"""
from __future__ import annotations

import time
import uuid

from ragagent.audit import AuditLog
from ragagent.config import Settings
from ragagent.generator import generate
from ragagent.ledger import BudgetExceeded, Ledger, LedgerEntry, estimate_tokens
from ragagent.models import (Answer, AnswerStatus, RetrievalHit, RetrievalPolicy,
                             SubQuery, ToolCallResult)
from ragagent.planner import plan
from ragagent.policies import PolicyRegistry
from ragagent.store import DocumentStore
from ragagent.sufficiency import follow_ups, sufficiency
from ragagent.tools import ToolRegistry, detect_tool_need


class QueryEngine:
    """Agentic RAG query engine (deterministic, offline, stdlib core)."""

    def __init__(self, settings: Settings, store: DocumentStore,
                 tools: ToolRegistry, ledger: Ledger, audit: AuditLog,
                 policies: PolicyRegistry):
        self.settings = settings
        self.store = store
        self.tools = tools
        self.ledger = ledger
        self.audit = audit
        self.policies = policies
        active = policies.active()
        if active is None:
            raise ValueError("policy registry has no ACTIVE policy")
        self.policy: RetrievalPolicy = active
        self.last_hits: list[RetrievalHit] = []

    def set_policy(self, version: str) -> RetrievalPolicy:
        """Use a different policy version for this engine (A/B eval).

        Does not change which policy is ACTIVE in the registry — only
        promotion does that.
        """
        policy = self.policies.get(version)
        if policy is None:
            raise ValueError(f"unknown policy version: {version}")
        self.policy = policy
        return policy

    # -- main entry point ------------------------------------------------
    def ask(self, question: str, tenant: str = "default") -> Answer:
        request_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        policy = self.policy
        est_tokens = estimate_tokens(question) + 400
        est_cost = (est_tokens / 1000.0 * self.settings.cost_per_1k_tokens
                    + self.settings.cost_per_retrieval_round)
        try:
            self.ledger.check(tenant, est_cost_usd=est_cost,
                              est_tokens=est_tokens)
        except BudgetExceeded as exc:
            self.audit.append("request_blocked", request_id, tenant,
                              {"question": question, "reason": str(exc)})
            return Answer(request_id=request_id, question=question,
                          status=AnswerStatus.BLOCKED, abstain_reason=str(exc),
                          policy_version=policy.version)

        sub_queries = plan(question, self.settings.default_max_subqueries)
        hits_by_chunk: dict[str, RetrievalHit] = {}
        tool_calls: list[ToolCallResult] = []
        open_queries = list(sub_queries)
        rounds = 0
        overall = 0.0
        per_query: dict[str, float] = {}
        for round_no in range(1, policy.max_rounds + 1):
            rounds = round_no
            for sq in open_queries:
                for hit in self.store.retrieve(
                        sq.text, top_k=policy.top_k,
                        vector_weight=policy.vector_weight):
                    prev = hits_by_chunk.get(hit.chunk.chunk_id)
                    if prev is None or hit.score > prev.score:
                        hits_by_chunk[hit.chunk.chunk_id] = hit
            hits = sorted(hits_by_chunk.values(),
                          key=lambda h: (-h.score, h.chunk.chunk_id))
            self.last_hits = hits
            overall, per_query = sufficiency(
                sub_queries, self.store,
                vector_weight=policy.vector_weight)
            self.audit.append("retrieval_round", request_id, tenant, {
                "round": round_no, "sub_queries": [sq.text for sq in open_queries],
                "hits": len(hits), "sufficiency": round(overall, 3)})
            if overall >= policy.sufficiency_threshold:
                break
            uncovered = [sq for sq in sub_queries
                         if per_query.get(sq.text, 0.0) < policy.sufficiency_threshold]
            next_queries = follow_ups(uncovered)
            if not next_queries or round_no == policy.max_rounds:
                break
            open_queries = next_queries

        hits = sorted(hits_by_chunk.values(),
                      key=lambda h: (-h.score, h.chunk.chunk_id))
        self.last_hits = hits

        # Tools fill gaps retrieval could not. Arithmetic / date /
        # definition sub-queries always try their tool: retrieved text
        # can state the inputs but never supply a computed sum, a
        # calendar fact, or a glossary definition. Factual sub-queries
        # only get a tool when still uncovered.
        if overall < policy.sufficiency_threshold or any(
                sq.kind in ("arithmetic", "date", "definition")
                for sq in sub_queries):
            uncovered = {sq.text for sq in sub_queries
                         if per_query.get(sq.text, 0.0) < policy.sufficiency_threshold}
            seen_tools: set[str] = set()
            for sq in sub_queries:
                if sq.kind not in ("arithmetic", "date", "definition") \
                        and sq.text not in uncovered:
                    continue
                detected = detect_tool_need(sq.text, self.tools.glossary)
                if detected is None:
                    continue
                name, args = detected
                key = f"{name}:{sorted(args.items())}"
                if key in seen_tools:
                    continue
                seen_tools.add(key)
                result = self.tools.call(name, args)
                tool_calls.append(result)
                self.audit.append("tool_call", request_id, tenant, {
                    "tool": name, "args": args, "ok": result.ok,
                    "result": result.result if result.ok else result.error})
            if any(r.ok for r in tool_calls):
                # A successful tool result is citable evidence; re-score
                # sufficiency with the gap treated as filled for the
                # sub-queries the tool answered.
                overall = max(overall, policy.sufficiency_threshold)

        status, text, citations, reason = generate(
            question, overall, hits, tool_calls,
            abstain_threshold=policy.abstain_threshold,
            min_evidence_score=policy.min_evidence_score,
            sub_queries=sub_queries)
        latency_ms = int((time.monotonic() - started) * 1000)
        tokens_in = estimate_tokens(question) + sum(
            estimate_tokens(h.chunk.text) for h in hits[:policy.top_k])
        tokens_out = estimate_tokens(text) if text else 0
        cost = ((tokens_in + tokens_out) / 1000.0
                * self.settings.cost_per_1k_tokens
                + rounds * self.settings.cost_per_retrieval_round
                + sum(r.cost_usd for r in tool_calls if r.ok))
        self.ledger.record(LedgerEntry(
            request_id=request_id, tenant=tenant, tokens_in=tokens_in,
            tokens_out=tokens_out, cost_usd=round(cost, 6),
            latency_ms=latency_ms, rounds=rounds,
            tool_calls=len(tool_calls), status=status.value))
        self.audit.append("answer", request_id, tenant, {
            "status": status.value, "sufficiency": round(overall, 3),
            "citations": [c.source for c in citations],
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_usd": round(cost, 6), "latency_ms": latency_ms,
            "abstain_reason": reason})
        return Answer(
            request_id=request_id, question=question, status=status,
            text=text, citations=citations, sub_queries=sub_queries,
            rounds=rounds, tool_calls=tool_calls, sufficiency=overall,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=round(cost, 6), latency_ms=latency_ms,
            abstain_reason=reason, policy_version=policy.version)
