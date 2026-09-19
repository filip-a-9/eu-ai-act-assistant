"""The one place in this repo that turns text into vectors.

It exists as its own module rather than living inside ``core/index.py``
because both sides of retrieval need it: indexing embeds 886 chunks offline,
and every query embeds one question at runtime. Putting the client in either
module would make the other depend on it for the wrong reason.

``Embedder`` is a Protocol, and ``OpenAIEmbedder`` takes its client as a field,
so tests substitute a fake and ``pytest -q`` never reaches the network.

The OpenAI SDK is used directly rather than through ``langchain-openai``:
one fewer layer between this file and the HTTP call, and nothing in LangGraph
requires LangChain's embedding wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# The API accepts up to 2,048 inputs per request, but a smaller batch keeps any
# single failure cheap to retry and the progress output meaningful. At 886
# chunks this is nine requests.
BATCH_SIZE = 100


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn a list of texts into a list of vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class OpenAIEmbedder:
    """Embeds through the OpenAI API, in batches, preserving input order.

    No retry logic here on purpose: the OpenAI client already retries
    transient failures, and a second layer of retries would obscure which one
    was responsible for a slow build.
    """

    client: Any  # openai.OpenAI, or a fake in tests
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model, input=batch)
            # Sorted by the index the API reports rather than trusting response
            # order. The documented behaviour is that they match, but a
            # mismatch here would file every vector under the wrong chunk and
            # raise nothing -- the cost of the sort is not worth the risk.
            for datum in sorted(response.data, key=lambda d: d.index):
                vectors.append(list(datum.embedding))
        return vectors


def openai_embedder(api_key: str, model: str) -> OpenAIEmbedder:
    """Build an embedder against the real API.

    Separate from the class so that importing this module never constructs a
    client, and so tests can build an ``OpenAIEmbedder`` without a key.
    """
    from openai import OpenAI

    return OpenAIEmbedder(client=OpenAI(api_key=api_key), model=model)
