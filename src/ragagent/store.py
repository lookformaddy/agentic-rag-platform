"""Hybrid retrieval store: deterministic hashed embeddings + keyword (BM25-style).

Chunks are built by packing sentences up to a word budget with a
one-sentence overlap, so a fact split across a sentence boundary stays
retrievable. Retrieval scores every chunk twice — cosine similarity on
the hashed embedding and a BM25-style keyword score — and combines them
with the active policy's vector weight. Scores are normalised to [0, 1].
"""
from __future__ import annotations

import math
import re

from ragagent.embedder import content_tokens, cosine, embed, tokenize
from ragagent.models import Chunk, Document, RetrievalHit

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CHUNK_WORDS = 60


def chunk_document(doc: Document, max_words: int = _CHUNK_WORDS) -> list[Chunk]:
    """Pack a document's sentences into overlapping chunks."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(doc.text.strip()) if s.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    words = 0
    index = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > max_words:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#c{index}", doc_id=doc.doc_id,
                text=" ".join(current), source=doc.source, chunk_index=index))
            index += 1
            current = [current[-1]] if current else []
            words = len(current[-1].split()) if current else 0
        current.append(sentence)
        words += n
    if current:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#c{index}", doc_id=doc.doc_id,
            text=" ".join(current), source=doc.source, chunk_index=index))
    return chunks


class DocumentStore:
    """In-memory hybrid retrieval over chunked documents."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._chunk_tokens: list[list[str]] = []
        self._doc_freq: dict[str, int] = {}
        self._n_chunks = 0

    def add_document(self, doc: Document) -> int:
        """Chunk and index one document; returns the chunk count."""
        new_chunks = chunk_document(doc)
        for chunk in new_chunks:
            tokens = content_tokens(chunk.text)
            self.chunks.append(chunk)
            self._vectors.append(embed(chunk.text, self.dim))
            self._chunk_tokens.append(tokens)
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._n_chunks = len(self.chunks)
        return len(new_chunks)

    def add_documents(self, docs: list[Document]) -> int:
        return sum(self.add_document(d) for d in docs)

    def __len__(self) -> int:
        return len(self.chunks)

    # -- scoring ---------------------------------------------------------
    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        if not df:
            # Unseen terms are maximally discriminative: they count in the
            # denominator so a query full of absent terms scores low.
            return math.log(self._n_chunks + 1.0)
        # Robertson / Sparck-Jones IDF, smoothed.
        return math.log((self._n_chunks - df + 0.5) / (df + 0.5) + 1.0)

    def _keyword_score(self, query_tokens: list[str], chunk_idx: int) -> float:
        """BM25-style score normalised to [0, 1] by the query's max."""
        if not query_tokens:
            return 0.0
        tokens = self._chunk_tokens[chunk_idx]
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        total_idf = 0.0
        score = 0.0
        for qt in set(query_tokens):
            idf = self._idf(qt)
            total_idf += idf
            tf = counts.get(qt, 0)
            if tf:
                score += idf * (tf / (tf + 1.0))
        if total_idf == 0.0:
            return 0.0
        return max(0.0, min(1.0, score / total_idf))

    def retrieve(self, query: str, top_k: int = 5,
                 vector_weight: float = 0.5) -> list[RetrievalHit]:
        """Hybrid retrieval: ``vector_weight`` * cosine + (1-w) * keyword."""
        if not self.chunks:
            return []
        qvec = embed(query, self.dim)
        qtokens = content_tokens(query)
        hits: list[RetrievalHit] = []
        for idx, chunk in enumerate(self.chunks):
            v = cosine(qvec, self._vectors[idx])
            k = self._keyword_score(qtokens, idx)
            score = vector_weight * v + (1.0 - vector_weight) * k
            hits.append(RetrievalHit(chunk=chunk, vector_score=v,
                                     keyword_score=k, score=score))
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:max(0, top_k)]

    def recall_at_k(self, query: str, expected_doc_ids: list[str],
                    top_k: int = 5, vector_weight: float = 0.5) -> float:
        """Fraction of expected doc ids present in the top-k hits."""
        if not expected_doc_ids:
            return 1.0
        hits = self.retrieve(query, top_k=top_k, vector_weight=vector_weight)
        found = {h.chunk.doc_id for h in hits}
        return sum(1 for d in expected_doc_ids if d in found) / len(expected_doc_ids)
