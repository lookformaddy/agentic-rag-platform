"""Query decomposition planner (deterministic, rule-based).

A question is decomposed into sub-queries the agent can retrieve against
independently:

- multi-part questions joined by "and" where the second clause is itself
  a question ("What is X and how much does Y cost?") split into parts;
- a keyword-only sub-query (stopwords stripped) for high-recall retrieval;
- the original question itself, always kept as the anchor sub-query.

Kinds are classified so the engine knows which follow-up tool can fill a
gap: ``arithmetic`` (numbers + total/sum/difference), ``date`` (when /
day / deadline), ``definition`` (what is / what does X mean), else
``factual``.
"""
from __future__ import annotations

import re

from ragagent.embedder import content_tokens
from ragagent.models import SubQuery

_QUESTION_WORDS = ("what", "how", "when", "where", "why", "which", "who",
                   "is", "are", "does", "do", "can", "could", "should")
_ARITHMETIC_HINTS = ("total", "sum", "add", "plus", "difference",
                     "altogether", "combined", "cost in total", "how much")
_DATE_HINTS = ("what day", "which day", "when is", "what date",
               "days until", "days between", "deadline")
_DEFINITION_HINTS = ("what is", "what does", "define", "meaning of",
                     "what do you mean by")


def classify(text: str) -> str:
    """Classify a sub-query by the kind of gap it may leave."""
    low = text.lower()
    if re.search(r"\d", low) and any(h in low for h in _ARITHMETIC_HINTS):
        return "arithmetic"
    if any(h in low for h in _DATE_HINTS):
        return "date"
    if any(h in low for h in _DEFINITION_HINTS):
        return "definition"
    return "factual"


def _split_multi_part(question: str) -> list[str]:
    """Split 'X and <question-word> Y?' into its question clauses."""
    parts = re.split(r"\s+and\s+", question.strip())
    if len(parts) < 2:
        return [question.strip()]
    # Only split when every clause after the first starts with a question word.
    for tail in parts[1:]:
        first = tail.strip().split(" ", 1)[0].lower().rstrip("?")
        if first not in _QUESTION_WORDS:
            return [question.strip()]
    out = [parts[0].strip().rstrip("?") + "?"]
    for tail in parts[1:]:
        tail = tail.strip().rstrip("?")
        out.append(tail[:1].upper() + tail[1:] + "?")
    return out


def plan(question: str, max_subqueries: int = 4) -> list[SubQuery]:
    """Decompose ``question`` into an ordered list of sub-queries."""
    question = question.strip()
    if not question:
        return []
    sub_queries: list[SubQuery] = []
    seen: set[str] = set()

    def add(text: str, kind: str | None = None, origin: str = "planner") -> None:
        key = text.strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        sub_queries.append(SubQuery(text=text.strip(),
                                    kind=kind or classify(text),
                                    origin=origin))

    for part in _split_multi_part(question):
        add(part)
    keywords = content_tokens(question)
    if len(keywords) >= 2:
        add(" ".join(keywords), kind="keyword")
    return sub_queries[:max_subqueries]
