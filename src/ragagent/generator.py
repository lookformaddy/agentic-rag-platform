"""Citation-attached generation with abstention (extractive, deterministic).

The generator never invents content: it selects the highest-scoring
retrieved sentences that share content tokens with the question, emits
them verbatim as the answer, and attaches each sentence's source
(document id + chunk, or ``tool:<name>``) as a citation. Tool results
are citable evidence too.

Selection is sub-query aware: each decomposed sub-query contributes its
best sentence first, so a multi-part question cannot fill the whole
answer from one document. A sentence must share at least two content
tokens with the question (one, if the question has only one) — a single
shared token such as "day" is not grounding.

Abstention is fail-closed: if sufficiency is below the policy's
abstain threshold, or no evidence clears the minimum evidence score,
the answer is an explicit abstention with a reason — never a guess.
"""
from __future__ import annotations

import re

from ragagent.embedder import content_tokens
from ragagent.models import (AnswerStatus, Evidence, RetrievalHit, SubQuery,
                             ToolCallResult)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _required_overlap(tokens: set[str]) -> int:
    return 2 if len(tokens) >= 2 else 1


def select_evidence(question: str, hits: list[RetrievalHit],
                    tool_results: list[ToolCallResult],
                    min_score: float, max_sentences: int = 3,
                    sub_queries: list[SubQuery] | None = None) -> list[Evidence]:
    """Pick the best evidence sentences from hits and tool results.

    Pass one grounds each sub-query with its best sentence; pass two
    fills any remaining slots with the best remaining sentences (at most
    one per chunk first, so one document cannot dominate). Tool evidence
    always leads: it fills a gap retrieval could not.
    """
    q_tokens = set(content_tokens(question))
    required = _required_overlap(q_tokens)
    candidates: list[tuple[Evidence, str, set[str]]] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.score < min_score:
            continue
        for sentence in _sentences(hit.chunk.text):
            key = sentence.lower()
            if key in seen:
                continue
            sent_tokens = set(content_tokens(sentence))
            if len(q_tokens & sent_tokens) < required:
                continue
            seen.add(key)
            candidates.append((Evidence(
                source=f"{hit.chunk.doc_id}#{hit.chunk.chunk_id}",
                text=sentence, score=hit.score),
                hit.chunk.chunk_id, sent_tokens))

    chosen: list[Evidence] = []
    chosen_keys: set[str] = set()

    def take(ev: Evidence) -> bool:
        if len(chosen) >= max_sentences or ev.text.lower() in chosen_keys:
            return False
        chosen.append(ev)
        chosen_keys.add(ev.text.lower())
        return True

    # Pass one: one best sentence per sub-query (highest token overlap
    # with that sub-query; hit order breaks ties).
    for sq in (sub_queries or []):
        if len(chosen) >= max_sentences:
            break
        sq_tokens = set(content_tokens(sq.text))
        need = _required_overlap(sq_tokens)
        best: Evidence | None = None
        best_overlap = -1
        for ev, _, sent_tokens in candidates:
            if ev.text.lower() in chosen_keys:
                continue
            overlap = len(sq_tokens & sent_tokens)
            if overlap >= need and overlap > best_overlap:
                best, best_overlap = ev, overlap
        if best is not None:
            take(best)
    # Pass two: best remaining, at most one per chunk first.
    used_chunks: set[str] = set()
    for ev, chunk_id, _ in candidates:
        if len(chosen) >= max_sentences:
            break
        if chunk_id in used_chunks:
            continue
        used_chunks.add(chunk_id)
        take(ev)
    for ev, _, _ in candidates:
        if len(chosen) >= max_sentences:
            break
        take(ev)

    tool_evidence = [Evidence(source=f"tool:{r.tool}", text=r.result, score=1.0)
                     for r in tool_results if r.ok]
    return (tool_evidence + chosen)[:max_sentences]


def generate(question: str, sufficiency: float, hits: list[RetrievalHit],
             tool_results: list[ToolCallResult], abstain_threshold: float,
             min_evidence_score: float,
             sub_queries: list[SubQuery] | None = None) -> tuple[AnswerStatus, str,
                                                 list[Evidence], str]:
    """Generate an answer or abstain. Returns (status, text, citations, reason)."""
    evidence = select_evidence(question, hits, tool_results,
                               min_evidence_score,
                               sub_queries=sub_queries)
    if sufficiency < abstain_threshold and not tool_results:
        return (AnswerStatus.ABSTAINED, "",
                [], "retrieval sufficiency %.2f is below the abstention "
                    "threshold %.2f; evidence is too thin to answer"
                    % (sufficiency, abstain_threshold))
    if not evidence:
        return (AnswerStatus.ABSTAINED, "",
                [], "no retrieved passage or tool result clears the minimum "
                    "evidence score %.2f" % min_evidence_score)
    lines = []
    for ev in evidence:
        lines.append(f"{ev.text} [{ev.source}]")
    return AnswerStatus.ANSWERED, " ".join(lines), evidence, ""
