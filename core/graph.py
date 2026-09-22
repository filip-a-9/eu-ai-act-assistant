"""The three-node graph: rewrite, retrieve, generate.

Flat by decision, and by the rule in CLAUDE.md. There are exactly two
conditional edges -- skip the rewrite on a first turn, and branch to a refusal
instead of shipping sources for an answer that cited none. No subgraph, no
agent loop, nothing that can call a tool twice.

This module is wiring only. Every prompt lives in ``core/generate.py`` and
every ranking decision in ``core/retrieve.py``, so the CLI, the eval and the
app cannot drift apart by going through different code.

Dependencies arrive as arguments rather than being constructed here, which is
what lets the tests build a graph over a six-chunk index with a fake embedder
and a canned generator, and never reach the network.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.embed import Embedder
from core.generate import Answer, Generator, answer, rewrite
from core.retrieve import Bm25Index, Hit, hybrid_search, log_query


class State(TypedDict, total=False):
    """What flows between the nodes.

    ``question`` is always what the user typed; ``rewritten`` is what was
    searched. Keeping both is what lets the query log record the pair, which is
    the only way to tell a bad rewrite apart from a bad retrieval afterwards.
    """

    question: str
    history: list[tuple[str, str]]
    rewritten: str | None
    hits: list[Hit]
    answer: Answer


def build_graph(
    *,
    collection: Any,
    bm25: Bm25Index,
    embedder: Embedder,
    generator: Generator,
    top_k: int,
    fusion_candidates: int,
    logs_dir: Path,
):
    """Compile the graph over the given retrievers, embedder and generator."""

    def rewrite_node(state: State) -> State:
        return {
            "rewritten": rewrite(generator, state["question"], state.get("history", []))
        }

    def retrieve_node(state: State) -> State:
        hits = hybrid_search(
            collection,
            bm25,
            embedder,
            _search_question(state),
            top_k,
            fusion_candidates,
        )
        # The one place a query is logged. Putting it in the node rather than
        # in the CLI means the app, the CLI and any future caller all log,
        # because none of them can retrieve without passing through here.
        log_query(state["question"], hits, logs_dir, state.get("rewritten"))
        return {"hits": hits}

    def generate_node(state: State) -> State:
        return {
            "answer": answer(generator, _search_question(state), state.get("hits", []))
        }

    def refuse_node(state: State) -> State:
        # Drop the retrieved chunks from a refusal. They did not support an
        # answer, and a refusal displayed above five provisions reads, to
        # anyone skimming, exactly like a cited one.
        return {"answer": replace(state["answer"], hits=())}

    builder = StateGraph(State)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("refuse", refuse_node)

    builder.add_conditional_edges(START, _first_step, ["rewrite", "retrieve"])
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_conditional_edges("generate", _after_generate, ["refuse", END])
    builder.add_edge("refuse", END)

    return builder.compile()


def _search_question(state: State) -> str:
    """What retrieval and generation see: the rewrite if there is one."""
    return state.get("rewritten") or state["question"]


def _first_step(state: State) -> str:
    """Skip the rewrite on a first turn -- there is nothing to resolve against."""
    return "rewrite" if state.get("history") else "retrieve"


def _after_generate(state: State) -> str:
    return "refuse" if state["answer"].refused else END


def run_turn(graph: Any, question: str, history: Sequence[tuple[str, str]]) -> Answer:
    """Run one turn and hand back its answer.

    A small seam so callers construct the state in one place: a caller that
    forgot ``history`` would silently lose the rewrite and look merely worse,
    not broken.
    """
    final = graph.invoke(
        {"question": question, "history": list(history), "rewritten": None}
    )
    return final["answer"]
