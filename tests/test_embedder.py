import math

from ragagent.embedder import content_tokens, cosine, embed, tokenize


def test_tokenize_lowercases_and_stems():
    assert tokenize("Costs $12.99, Points!") == ["cost", "12", "99", "point"]


def test_content_tokens_drop_stopwords():
    tokens = content_tokens("What is the return window?")
    assert tokens == ["return", "window"]


def test_embed_is_deterministic():
    assert embed("return window") == embed("return window")
    assert embed("return window") != embed("shipping cost")


def test_embed_is_unit_norm():
    vec = embed("the quick brown fox")
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-9)


def test_embed_empty_text_is_zero_vector():
    assert embed("") == [0.0] * 128
    assert embed("hello", dim=16) != embed("hello", dim=32)


def test_cosine_self_is_one():
    vec = embed("return window")
    assert math.isclose(cosine(vec, vec), 1.0, rel_tol=1e-9)


def test_cosine_unrelated_is_low():
    a = embed("return window refund exchange")
    b = embed("quantum chromodynamics lattice gauge")
    assert cosine(a, b) < 0.5
