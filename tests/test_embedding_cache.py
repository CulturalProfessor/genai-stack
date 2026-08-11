import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.embedding_cache import CachedEmbeddings, wrap


class FakeEmbeddings:
    """Counts what it was asked to embed, and returns a vector per text."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.5, -1.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    @property
    def embedded(self):
        return [text for call in self.calls for text in call]


def build(tmp_path, inner=None, model_key="test:3"):
    return CachedEmbeddings(
        inner or FakeEmbeddings(), model_key=model_key, path=tmp_path / "cache.sqlite"
    )


def test_second_call_does_not_reach_the_model(tmp_path):
    inner = FakeEmbeddings()
    cache = build(tmp_path, inner)

    first = cache.embed_documents(["alpha", "beta"])
    second = cache.embed_documents(["alpha", "beta"])

    assert first == second
    assert inner.embedded == ["alpha", "beta"]
    assert cache.stats()["hits"] == 2


def test_only_the_new_texts_are_embedded(tmp_path):
    inner = FakeEmbeddings()
    cache = build(tmp_path, inner)

    cache.embed_documents(["alpha"])
    cache.embed_documents(["alpha", "gamma"])

    assert inner.embedded == ["alpha", "gamma"]


def test_order_is_preserved_when_a_batch_is_partly_cached(tmp_path):
    cache = build(tmp_path)
    cache.embed_documents(["bb"])

    vectors = cache.embed_documents(["a", "bb", "ccc"])

    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


def test_a_repeated_text_in_one_batch_costs_one_call(tmp_path):
    inner = FakeEmbeddings()
    cache = build(tmp_path, inner)

    vectors = cache.embed_documents(["same", "other", "same"])

    assert inner.embedded == ["same", "other"]
    assert vectors[0] == vectors[2]


def test_vectors_survive_a_reopen(tmp_path):
    inner = FakeEmbeddings()
    build(tmp_path, inner).embed_documents(["alpha"])

    reopened = build(tmp_path, inner)
    assert reopened.embed_documents(["alpha"]) == [[5.0, 0.5, -1.5]]
    assert inner.embedded == ["alpha"]


def test_another_model_does_not_read_the_first_ones_vectors(tmp_path):
    """The key covers the model, so switching it cannot return a stale vector
    of the wrong meaning - or, worse, the wrong width."""
    inner = FakeEmbeddings()
    build(tmp_path, inner, model_key="openai:1536").embed_documents(["alpha"])
    build(tmp_path, inner, model_key="ollama:4096").embed_documents(["alpha"])

    assert inner.embedded == ["alpha", "alpha"]


def test_query_and_document_share_the_cache(tmp_path):
    inner = FakeEmbeddings()
    cache = build(tmp_path, inner)

    cache.embed_documents(["alpha"])
    assert cache.embed_query("alpha") == [5.0, 0.5, -1.5]
    assert inner.embedded == ["alpha"]


def test_wrap_disabled_hands_back_the_original(tmp_path):
    inner = FakeEmbeddings()
    assert wrap(inner, "openai", 1536, path=tmp_path / "c.sqlite", enabled=False) is inner
