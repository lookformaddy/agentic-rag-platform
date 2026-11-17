"""Deterministic hashed embeddings (pure stdlib).

No model downloads, no network, no numpy: each token is hashed with
SHA-256 into one of ``dim`` buckets, counts are accumulated, and the
vector is L2-normalised. Identical text always yields the identical
vector, so retrieval is reproducible and testable.
"""
from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "with", "by", "at", "from",
    "as", "it", "its", "this", "that", "these", "those", "what", "which",
    "who", "whom", "how", "when", "where", "why", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "you", "we", "they",
    "he", "she", "my", "your", "our", "their", "if", "then", "than",
    "so", "such", "no", "not", "but", "about", "into", "over", "under",
    "after", "before", "between", "per", "each", "any", "all", "there",
    "here", "have", "has", "had", "get", "gets", "me", "us", "them",
    "many", "much", "few", "more", "most",
})


def _stem(token: str) -> str:
    """Minimal suffix stemming so 'costs' matches 'cost' and
    'points' matches 'point'. Applied identically to queries and chunks."""
    if len(token) > 4 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 3 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("es"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, lightly stemmed."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed (for keyword retrieval / planning)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic hashed embedding of ``text`` into ``dim`` buckets."""
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (already normalised)."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))
