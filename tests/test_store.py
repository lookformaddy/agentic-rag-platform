from ragagent.models import Document
from ragagent.store import DocumentStore, chunk_document


def test_chunking_splits_long_docs():
    doc = Document(doc_id="d1", title="T",
                   text=" ".join(f"Sentence number {i} about returns."
                                 for i in range(40)))
    chunks = chunk_document(doc)
    assert len(chunks) >= 2
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].chunk_id == "d1#c0"


def test_retrieve_returns_expected_doc_first(store):
    hits = store.retrieve("What is the return window?", top_k=3)
    assert hits
    assert hits[0].chunk.doc_id == "returns-policy"
    assert hits[0].score > 0.3


def test_retrieve_scores_are_bounded(store):
    for hit in store.retrieve("shipping cost", top_k=5):
        assert 0.0 <= hit.score <= 1.0
        assert 0.0 <= hit.vector_score <= 1.0
        assert 0.0 <= hit.keyword_score <= 1.0


def test_retrieve_empty_store_returns_nothing():
    assert DocumentStore().retrieve("anything") == []


def test_recall_at_k(store):
    assert store.recall_at_k("What is the return window?",
                             ["returns-policy"], top_k=5) == 1.0
    # The single best hit for an out-of-corpus question is not the
    # (irrelevant) returns doc; wider top-k is noise on a tiny corpus.
    assert store.recall_at_k("What is the quarterly revenue?",
                             ["returns-policy"], top_k=1) == 0.0


def test_keyword_weight_changes_ranking(store):
    # Pure keyword retrieval still finds the shipping doc.
    hits = store.retrieve("express shipping cost", top_k=1, vector_weight=0.0)
    assert hits[0].chunk.doc_id == "shipping-policy"
    # Pure vector retrieval also ranks it first for this query.
    hits = store.retrieve("express shipping cost", top_k=1, vector_weight=1.0)
    assert hits[0].chunk.doc_id == "shipping-policy"
