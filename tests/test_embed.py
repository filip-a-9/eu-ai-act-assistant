"""Contract for the embedding seam.

No test here touches the network. ``OpenAIEmbedder`` takes its client as a
field precisely so these tests can hand it a fake -- that seam is the whole
reason the module exists separately from ``core/index.py``.

The assertions worth reading twice are the ordering ones. A batching bug that
returns 250 correct vectors in the wrong order produces an index where every
chunk is filed under someone else's meaning, and absolutely nothing raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from core.embed import BATCH_SIZE, Embedder, OpenAIEmbedder

# ---------------------------------------------------------------------------
# A fake standing in for openai.OpenAI, mimicking only the response shape used
# ---------------------------------------------------------------------------


@dataclass
class _FakeDatum:
    index: int
    embedding: list[float]


@dataclass
class _FakeResponse:
    data: list[_FakeDatum]


@dataclass
class _FakeEmbeddings:
    calls: list[dict] = field(default_factory=list)
    shuffle: bool = False

    def create(self, *, model: str, input: list[str]):
        self.calls.append({"model": model, "input": list(input)})
        # One dimension per input position is enough to identify a vector:
        # the tests only need to tell vectors apart, not measure distance.
        data = [
            _FakeDatum(index=i, embedding=[float(len(text)), float(i)])
            for i, text in enumerate(input)
        ]
        if self.shuffle:
            data = list(reversed(data))
        return _FakeResponse(data=data)


@dataclass
class _FakeClient:
    embeddings: _FakeEmbeddings = field(default_factory=_FakeEmbeddings)


@pytest.fixture
def fake_client():
    return _FakeClient()


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_returns_one_vector_per_input(fake_client):
    embedder = OpenAIEmbedder(client=fake_client, model="text-embedding-3-small")
    vectors = embedder.embed(["alpha", "beta", "gamma"])
    assert len(vectors) == 3


def test_an_empty_input_makes_no_api_call(fake_client):
    # Cheap, but the alternative is a 400 from the API on an empty corpus --
    # an unhelpful failure a long way from the cause.
    embedder = OpenAIEmbedder(client=fake_client, model="text-embedding-3-small")
    assert embedder.embed([]) == []
    assert fake_client.embeddings.calls == []


def test_the_configured_model_reaches_the_client(fake_client):
    # CLAUDE.md forbids model names as literals in logic; this proves the one
    # from config is the one actually used.
    embedder = OpenAIEmbedder(client=fake_client, model="text-embedding-3-large")
    embedder.embed(["alpha"])
    assert fake_client.embeddings.calls[0]["model"] == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_a_small_input_is_sent_as_one_request(fake_client):
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    embedder.embed(["alpha", "beta"])
    assert len(fake_client.embeddings.calls) == 1


def test_a_large_input_is_split_into_batches(fake_client):
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    texts = [f"chunk {i}" for i in range(BATCH_SIZE * 2 + 5)]
    embedder.embed(texts)
    sizes = [len(call["input"]) for call in fake_client.embeddings.calls]
    assert sizes == [BATCH_SIZE, BATCH_SIZE, 5]


def test_batching_preserves_input_order_across_batch_boundaries(fake_client):
    # The failure this guards against is silent: 886 correct vectors attached
    # to the wrong 886 chunks.
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    texts = [f"{'x' * i}" for i in range(BATCH_SIZE + 10)]
    vectors = embedder.embed(texts)
    # The fake encodes len(text) in dimension 0, so the vector at position i
    # must report length i no matter which batch carried it.
    assert [v[0] for v in vectors] == [float(i) for i in range(len(texts))]


def test_every_input_is_sent_exactly_once(fake_client):
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    texts = [f"chunk {i}" for i in range(BATCH_SIZE + 1)]
    embedder.embed(texts)
    sent = [text for call in fake_client.embeddings.calls for text in call["input"]]
    assert sent == texts


# ---------------------------------------------------------------------------
# Ordering within a response
# ---------------------------------------------------------------------------


def test_vectors_are_ordered_by_the_index_the_api_reports(fake_client):
    # The API documents that `data` comes back in request order. Sorting on the
    # `index` field anyway costs nothing and removes the need to trust that.
    fake_client.embeddings.shuffle = True
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


def test_openai_embedder_satisfies_the_embedder_protocol(fake_client):
    embedder = OpenAIEmbedder(client=fake_client, model="m")
    assert isinstance(embedder, Embedder)


def test_a_hand_written_stub_satisfies_the_embedder_protocol():
    # If this fails, tests elsewhere cannot substitute the embedder and the
    # no-network rule becomes unenforceable.
    class Stub:
        def embed(self, texts):
            return [[0.0] for _ in texts]

    assert isinstance(Stub(), Embedder)
