"""Contract for retrieval.

These tests assert wiring and ranking *mechanics* -- that the top_k asked for
is the top_k returned, that scores descend, that a Chroma distance becomes a
similarity, that a Hit carries the citation the answer will quote. They do not
assert retrieval quality; "is Article 5 the right answer for this question"
is a judgement over a question set and belongs in ``evals/run_eval.py``.

The ordering assertions are only meaningful because ``FakeEmbedder`` is a bag
of words rather than a hash: "which practices are prohibited" genuinely lands
nearest the prohibited-practices chunk, so a broken query path shows up here
instead of passing on an arbitrary but stable ordering.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from core.index import build, open_collection
from core.retrieve import Hit, log_query, search


@pytest.fixture
def collection(built_index):
    return open_collection(built_index)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_returns_exactly_the_number_of_hits_asked_for(collection, fake_embedder):
    assert len(search(collection, fake_embedder, "prohibited practices", 3)) == 3


def test_a_top_k_larger_than_the_corpus_returns_everything(
    collection, fake_embedder, tiny_corpus
):
    # Chroma clamps rather than erroring; this pins that, because the eval
    # sweeps top_k and a six-chunk fixture is easy to over-ask.
    hits = search(collection, fake_embedder, "prohibited practices", 50)
    assert len(hits) == len(tiny_corpus)


def test_a_top_k_of_zero_returns_nothing_and_does_not_error(collection, fake_embedder):
    assert search(collection, fake_embedder, "prohibited practices", 0) == []


def test_an_empty_collection_returns_no_hits(tmp_path, fake_embedder):
    build([], fake_embedder, tmp_path / "empty")
    collection = open_collection(tmp_path / "empty")
    assert search(collection, fake_embedder, "prohibited practices", 5) == []


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_hits_are_ordered_best_first(collection, fake_embedder):
    scores = [hit.score for hit in search(collection, fake_embedder, "risk", 6)]
    assert scores == sorted(scores, reverse=True)


def test_rank_is_one_based_and_sequential(collection, fake_embedder):
    hits = search(collection, fake_embedder, "risk management", 4)
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]


def test_the_score_is_one_minus_the_cosine_distance(collection, fake_embedder):
    # Chroma reports distance; an answer surface wants similarity. Getting the
    # direction wrong would reverse the ranking, which is worth pinning.
    question = "prohibited practices"
    hits = search(collection, fake_embedder, question, 3)
    raw = collection.query(
        query_embeddings=fake_embedder.embed([question]),
        n_results=3,
        include=["distances"],
    )
    # strict: same reason as in test_index -- a short list here would quietly
    # check fewer conversions than there are hits, or none at all.
    for hit, distance in zip(hits, raw["distances"][0], strict=True):
        assert hit.score == pytest.approx(1.0 - distance)


def test_an_identical_question_scores_near_one(collection, fake_embedder):
    # Sanity on the direction of the conversion: cosine distance 0 is a
    # perfect match, so the score must be near 1, not near 0.
    hits = search(collection, fake_embedder, "penalties fines", 1)
    assert hits[0].score > 0.5


# ---------------------------------------------------------------------------
# Ranking that reflects the question, not the corpus order
# ---------------------------------------------------------------------------


def test_a_question_about_prohibitions_ranks_the_prohibitions_chunk_first(
    collection, fake_embedder
):
    hits = search(collection, fake_embedder, "which AI practices are prohibited", 3)
    assert hits[0].id == "art_5.para_1"


def test_a_question_about_fines_ranks_the_penalties_chunk_first(
    collection, fake_embedder
):
    hits = search(collection, fake_embedder, "what administrative fines apply", 3)
    assert hits[0].id == "art_99.para_3"


def test_a_question_about_risk_management_ranks_article_9_first(
    collection, fake_embedder
):
    # Deliberately close to Article 6, which also talks about high risk: this
    # is the pair most likely to swap if the query vector is wrong.
    hits = search(collection, fake_embedder, "risk management system lifecycle", 3)
    assert hits[0].id == "art_9.para_1"


def test_different_questions_produce_different_orderings(collection, fake_embedder):
    # Guards the failure where the question is ignored and the corpus order is
    # returned every time -- which every test above would still pass if the
    # first chunk happened to be the expected one.
    first = [hit.id for hit in search(collection, fake_embedder, "prohibited", 6)]
    second = [hit.id for hit in search(collection, fake_embedder, "penalties", 6)]
    assert first != second


# ---------------------------------------------------------------------------
# What a Hit carries
# ---------------------------------------------------------------------------


def test_a_hit_carries_the_citation_and_the_source_url(collection, fake_embedder):
    # Every claim cites article_no, so a Hit that cannot produce one is
    # unusable downstream.
    for hit in search(collection, fake_embedder, "prohibited practices", 5):
        assert hit.article_no
        assert hit.source_url


def test_a_hit_carries_the_clean_law_not_the_embedded_text(
    collection, fake_embedder, tiny_corpus
):
    texts = {record["id"]: record["text"] for record in tiny_corpus}
    for hit in search(collection, fake_embedder, "prohibited practices", 5):
        assert hit.text == texts[hit.id]
        assert not hit.text.startswith(hit.article_no)


def test_a_hit_carries_every_field_the_answer_surface_needs(collection, fake_embedder):
    hit = search(collection, fake_embedder, "prohibited practices", 1)[0]
    assert isinstance(hit, Hit)
    for attribute in (
        "rank",
        "id",
        "score",
        "article_no",
        "title",
        "chapter",
        "kind",
        "source_url",
        "text",
    ):
        assert hasattr(hit, attribute)


def test_hits_are_immutable(collection, fake_embedder):
    hit = search(collection, fake_embedder, "prohibited practices", 1)[0]
    # See test_config_is_frozen: the point is that assignment is refused
    # because the dataclass is frozen, not that it happens to fail somehow.
    with pytest.raises(FrozenInstanceError):
        hit.score = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# A response whose parallel lists disagree
# ---------------------------------------------------------------------------


class TruncatedResponseCollection:
    """A Chroma stand-in returning one fewer metadata than it returns ids.

    Chroma builds the four lists together, so a real response cannot look like
    this today. The stub exists to pin what happens if that ever stops holding
    -- not to stand in for the collection in any other test.
    """

    def __init__(self, hits: int):
        self.hits = hits

    def query(self, **_kwargs):
        metadata = {
            "article_no": "Article 5(1)",
            "title": "Prohibited AI practices",
            "chapter": "Chapter II",
            "kind": "article",
            "source_url": "https://example.invalid/art5",
        }
        return {
            "ids": [[f"art_5.para_{n}" for n in range(self.hits)]],
            "documents": [["the law" for _ in range(self.hits)]],
            "metadatas": [[metadata for _ in range(self.hits - 1)]],
            "distances": [[0.1 for _ in range(self.hits)]],
        }


def test_a_response_missing_a_metadata_entry_raises_instead_of_dropping_a_hit(
    fake_embedder,
):
    # zip stops at the shortest input, so without strict= this returns four
    # Hits for five retrieved chunks and nothing anywhere reports it. In a
    # citation-grounded assistant a dropped hit is a dropped citation, and the
    # answer built from the survivors still looks complete.
    #
    # Only the stable clause of CPython's message is matched. The full text
    # names the argument position, which would change if the zip arguments
    # were ever reordered; "shorter" is what distinguishes "refused to
    # truncate" from some other ValueError out of the query path.
    with pytest.raises(ValueError, match="shorter"):
        search(TruncatedResponseCollection(hits=5), fake_embedder, "prohibited", 5)


# ---------------------------------------------------------------------------
# The query log
# ---------------------------------------------------------------------------


def test_the_log_records_the_question_the_ids_and_the_scores(
    tmp_path, collection, fake_embedder
):
    hits = search(collection, fake_embedder, "prohibited practices", 3)
    log_query("prohibited practices", hits, tmp_path)
    entry = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert entry["question"] == "prohibited practices"
    assert entry["chunk_ids"] == [hit.id for hit in hits]
    assert entry["scores"] == [hit.score for hit in hits]


def test_the_log_carries_a_rewritten_question_slot(tmp_path, collection, fake_embedder):
    # CLAUDE.md requires the rewritten question in the log. No rewrite node
    # exists yet, so the field is present and null rather than absent -- a
    # missing key would make the log format change when the graph lands.
    hits = search(collection, fake_embedder, "prohibited practices", 1)
    log_query("prohibited practices", hits, tmp_path)
    entry = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert "rewritten" in entry and entry["rewritten"] is None


def test_the_log_records_a_rewritten_question_when_given_one(
    tmp_path, collection, fake_embedder
):
    hits = search(collection, fake_embedder, "prohibited practices", 1)
    log_query("what is banned", hits, tmp_path, rewritten="prohibited practices")
    entry = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert entry["rewritten"] == "prohibited practices"


def test_the_log_appends_rather_than_overwriting(tmp_path, collection, fake_embedder):
    hits = search(collection, fake_embedder, "prohibited practices", 1)
    log_query("first", hits, tmp_path)
    log_query("second", hits, tmp_path)
    lines = (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question"] for line in lines] == ["first", "second"]


def test_the_log_directory_is_created_if_it_does_not_exist(
    tmp_path, collection, fake_embedder
):
    hits = search(collection, fake_embedder, "prohibited practices", 1)
    log_query("first", hits, tmp_path / "nested" / "logs")
    assert (tmp_path / "nested" / "logs" / "queries.jsonl").exists()


def test_the_log_records_a_refusal_shaped_empty_result(tmp_path):
    # Retrieval returning nothing is the case that must end in a refusal, so
    # it is exactly the case worth having in the log.
    log_query("something unrelated", [], tmp_path)
    entry = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert entry["chunk_ids"] == [] and entry["scores"] == []


def test_the_log_is_one_json_object_per_line(tmp_path, collection, fake_embedder):
    hits = search(collection, fake_embedder, "prohibited practices", 2)
    log_query("a", hits, tmp_path)
    log_query("b", hits, tmp_path)
    for line in (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_the_log_never_contains_an_api_key(tmp_path, collection, fake_embedder):
    # The config object hides its key from repr(); this is the same guarantee
    # checked at the other end, where the log is actually written.
    hits = search(collection, fake_embedder, "prohibited practices", 2)
    log_query("prohibited practices", hits, tmp_path)
    assert "sk-" not in (tmp_path / "queries.jsonl").read_text(encoding="utf-8")
