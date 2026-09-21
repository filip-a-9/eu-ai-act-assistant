"""Contract for the graph.

The graph is wiring, so these tests assert what the wiring causes: that a
first turn does not pay for a rewrite, that a follow-up is retrieved on the
rewritten question rather than on "what about them?", that every turn lands in
the query log, and that a refusal ships no sources.

Whether the rewrite is any *good* is not assertable here -- the generator is a
fake returning a canned string. That is the eval's question.
"""

from __future__ import annotations

import json

import pytest

from core.generate import REFUSAL_TEXT
from core.graph import build_graph, run_turn
from core.index import open_collection
from core.retrieve import LOG_FILENAME

ANSWER = "Social scoring is prohibited [Article 5(1)]."
UNCITED = "Yes, you may do that."
REWRITTEN = "What are the penalties for prohibited practices?"


@pytest.fixture
def graph_for(tmp_path, built_index, fake_embedder):
    """Build a graph over the tiny index, with the generator supplied per test."""
    collection = open_collection(built_index)
    logs_dir = tmp_path / "logs"

    def _build(generator, top_k=3):
        return (
            build_graph(
                collection=collection,
                embedder=fake_embedder,
                generator=generator,
                top_k=top_k,
                logs_dir=logs_dir,
            ),
            logs_dir,
        )

    return _build


def log_lines(logs_dir):
    path = logs_dir / LOG_FILENAME
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_the_graph_holds_only_the_four_nodes_claude_md_allows(
    graph_for, exploding_generator
):
    # A guard on flatness: rewrite, retrieve, generate and the refusal branch.
    # An agent loop or a subgraph arriving later shows up here.
    graph, _ = graph_for(exploding_generator)
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"rewrite", "retrieve", "generate", "refuse"}


# ---------------------------------------------------------------------------
# Skip rewrite on the first turn
# ---------------------------------------------------------------------------


def test_a_first_turn_is_logged_with_no_rewrite(graph_for, fake_generator):
    graph, logs_dir = graph_for(fake_generator(ANSWER))

    run_turn(graph, "which practices are prohibited?", history=[])

    assert log_lines(logs_dir)[0]["rewritten"] is None


def test_a_follow_up_is_retrieved_on_the_rewritten_question(graph_for, fake_generator):
    # "what about them?" carries no searchable term at all. If the top hit is
    # the penalties chunk, retrieval ran on the rewrite and not on the
    # follow-up as typed.
    graph, logs_dir = graph_for(fake_generator(REWRITTEN, ANSWER))
    history = [("which practices are prohibited?", ANSWER)]

    answer = run_turn(graph, "what about them?", history=history)

    assert log_lines(logs_dir)[0]["rewritten"] == REWRITTEN
    assert answer.hits[0].id == "art_99.para_3"


def test_the_log_records_the_question_as_the_user_typed_it(graph_for, fake_generator):
    graph, logs_dir = graph_for(fake_generator(REWRITTEN, ANSWER))
    history = [("which practices are prohibited?", ANSWER)]

    run_turn(graph, "what about them?", history=history)

    assert log_lines(logs_dir)[0]["question"] == "what about them?"


def test_every_turn_appends_one_log_line(graph_for, fake_generator):
    graph, logs_dir = graph_for(fake_generator(ANSWER, ANSWER))

    run_turn(graph, "which practices are prohibited?", history=[])
    run_turn(graph, "what are the penalties?", history=[])

    assert len(log_lines(logs_dir)) == 2


# ---------------------------------------------------------------------------
# Refuse vs answer
# ---------------------------------------------------------------------------


def test_an_answered_turn_carries_the_provisions_it_cited(graph_for, fake_generator):
    graph, _ = graph_for(fake_generator(ANSWER))

    answer = run_turn(graph, "which practices are prohibited?", history=[])

    assert answer.refused is False
    assert [hit.id for hit in answer.cited_hits] == ["art_5.para_1"]


def test_a_refused_turn_ships_no_sources(graph_for, fake_generator):
    # The refusal branch exists for this: a refusal that still displayed five
    # retrieved provisions would look like a cited answer to anyone skimming.
    graph, _ = graph_for(fake_generator(UNCITED))

    answer = run_turn(graph, "may I score my customers?", history=[])

    assert answer.text == REFUSAL_TEXT
    assert answer.hits == ()
