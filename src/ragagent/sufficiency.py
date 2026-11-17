"""Retrieval sufficiency scoring and follow-up query generation.

Self-critique step of the agent loop: after each retrieval round, every
sub-query is scored by the best hybrid score any chunk achieves against
it (top-1 hybrid retrieval for that sub-query). The overall sufficiency
is the *weakest* sub-query's coverage — an answer is only as grounded
as its least-covered part. Using the hybrid score (not raw token
overlap) matters: a question about an absent topic can share brand
tokens with a chunk yet still score low, which is exactly what must
trigger abstention.

Uncovered sub-queries drive deterministic follow-up reformulations:
keyword-only form, synonym expansion, and a definition-seeking form.
"""
from __future__ import annotations

from ragagent.embedder import content_tokens
from ragagent.models import SubQuery
from ragagent.store import DocumentStore

# Small built-in synonym map for follow-up expansion (generic retail /
# general-English terms only — no employer vocabulary).
SYNONYMS: dict[str, list[str]] = {
    "return": ["refund", "exchange"],
    "returns": ["refund", "exchange"],
    "cost": ["price", "fee"],
    "price": ["cost", "fee"],
    "shipping": ["delivery", "postage"],
    "delivery": ["shipping"],
    "warranty": ["guarantee", "coverage"],
    "discount": ["coupon", "offer", "promotion"],
    "store": ["shop", "location"],
    "hours": ["opening", "times"],
    "points": ["rewards", "credit"],
    "jacket": ["coat", "outerwear"],
    "boots": ["footwear", "shoes"],
}


def coverage_for(sub_query: SubQuery, store: DocumentStore,
                 vector_weight: float = 0.5) -> float:
    """Best hybrid retrieval score for one sub-query (top-1)."""
    hits = store.retrieve(sub_query.text, top_k=1, vector_weight=vector_weight)
    return hits[0].score if hits else 0.0


def sufficiency(sub_queries: list[SubQuery], store: DocumentStore,
                vector_weight: float = 0.5) -> tuple[float, dict[str, float]]:
    """Return (overall, per-sub-query coverage). Overall = weakest link."""
    per: dict[str, float] = {}
    for sq in sub_queries:
        per[sq.text] = coverage_for(sq, store, vector_weight)
    overall = min(per.values()) if per else 0.0
    return overall, per


def follow_ups(uncovered: list[SubQuery]) -> list[SubQuery]:
    """Deterministic reformulations for sub-queries that lack evidence."""
    out: list[SubQuery] = []
    seen: set[str] = set()
    for sq in uncovered:
        tokens = content_tokens(sq.text)
        if tokens:
            kw = " ".join(tokens)
            if kw.lower() not in seen:
                seen.add(kw.lower())
                out.append(SubQuery(text=kw, kind="keyword", origin="followup"))
        expansions: list[str] = []
        for t in tokens:
            expansions.extend(SYNONYMS.get(t, []))
        if expansions:
            expanded = " ".join(tokens + expansions)
            if expanded.lower() not in seen:
                seen.add(expanded.lower())
                out.append(SubQuery(text=expanded, kind="keyword",
                                    origin="followup"))
        if sq.kind == "definition" and tokens:
            term = tokens[-1]
            text = f"definition of {term}"
            if text not in seen:
                seen.add(text)
                out.append(SubQuery(text=text, kind="definition",
                                    origin="followup"))
    return out
