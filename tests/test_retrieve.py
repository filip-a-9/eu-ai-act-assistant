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
from core.retrieve import Hit, fuse, hybrid_search, log_query, search


@pytest.fixture
def collection(built_index):
    return open_collection(built_index)


@pytest.fixture
def empty_collection(tmp_path, fake_embedder):
    """A real but empty Chroma collection: what a half-built index looks like.

    Also the only way to make the dense side contribute nothing on purpose.
    ``FakeEmbedder`` maps a question of purely out-of-vocabulary words to the
    constant dimension alone, which a chunk of purely out-of-vocabulary words
    matches at 1.0 -- so "the embedder cannot see this" cannot be staged by
    choosing words. Emptying the collection states it directly.
    """
    build([], fake_embedder, tmp_path / "empty")
    return open_collection(tmp_path / "empty")


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


# ---------------------------------------------------------------------------
# The lexical half
# ---------------------------------------------------------------------------


def test_bm25_finds_a_chunk_by_a_term_the_embedder_cannot_represent(bm25_index):
    # "sandbox" is outside FakeEmbedder's VOCABULARY, so the dense side maps
    # this question to no direction at all. This is the miniature of the real
    # problem: an exact, rare token that an embedding blurs away.
    found = bm25_index.search("regulatory sandbox", 5)
    assert found[0][0]["id"] == "art_57.para_1"


def test_bm25_does_not_return_chunks_that_share_no_terms(bm25_index):
    # A zero-score chunk matched nothing. Returning it as a candidate would
    # hand it fusion credit for merely being in the list, which is how a
    # lexical retriever poisons a ranking it had no opinion about.
    found = bm25_index.search("regulatory sandbox", 7)
    assert [record["id"] for record, _ in found] == ["art_57.para_1"]


def test_bm25_returns_no_candidates_when_nothing_matches(bm25_index):
    assert bm25_index.search("xyzzy plugh", 5) == []


def test_bm25_over_an_empty_corpus_returns_nothing_rather_than_raising():
    # A fresh clone has no chunks.jsonl content until build_chunks.py runs,
    # and BM25Okapi divides by the average document length on construction.
    # Retrieval degrading to dense-only is recoverable; a ZeroDivisionError
    # at import time is not.
    from core.retrieve import Bm25Index

    assert Bm25Index.from_records([]).search("anything", 5) == []


def test_bm25_ranks_are_one_based_and_sequential(bm25_index):
    found = bm25_index.search("high risk system", 4)
    assert [rank for _, rank in found] == list(range(1, len(found) + 1))


def test_bm25_matches_the_citation_header_not_only_the_body(bm25_index):
    # The header is in embed_text and nowhere in text. "Annex III" is the
    # canonical case: the label is the thing a person types.
    found = bm25_index.search("Annex III", 3)
    assert found[0][0]["id"] == "anx_III.sec_1"


# ---------------------------------------------------------------------------
# Fusion mechanics -- fuse() is exercised directly, with ranks chosen by hand,
# so the expected score is arithmetic rather than a restatement of the code.
# ---------------------------------------------------------------------------


def _dense_hit(chunk_id, rank, score=0.5):
    return Hit(
        rank=rank,
        id=chunk_id,
        score=score,
        article_no=chunk_id,
        title="",
        chapter="",
        kind="article",
        source_url="https://example.invalid",
        text="",
        dense_rank=rank,
        dense_score=score,
    )


def _lexical(chunk_id, rank):
    return (
        {
            "id": chunk_id,
            "article_no": chunk_id,
            "title": "",
            "chapter": "",
            "kind": "article",
            "source_url": "https://example.invalid",
            "text": "",
        },
        rank,
    )


def test_the_fused_score_sums_the_reciprocal_of_each_rank():
    fused = fuse([_dense_hit("a", 1)], [_lexical("a", 3)], 1)
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_a_chunk_only_the_dense_side_found_carries_no_bm25_rank():
    fused = fuse([_dense_hit("a", 1)], [], 1)
    assert fused[0].dense_rank == 1 and fused[0].bm25_rank is None
    assert fused[0].score == pytest.approx(1 / 61)


def test_a_chunk_only_bm25_found_carries_no_dense_rank_or_score():
    fused = fuse([], [_lexical("a", 1)], 1)
    assert fused[0].bm25_rank == 1
    assert fused[0].dense_rank is None and fused[0].dense_score is None


def test_a_chunk_both_sides_found_appears_once():
    fused = fuse([_dense_hit("a", 1)], [_lexical("a", 1)], 5)
    assert [hit.id for hit in fused] == ["a"]


def test_agreement_between_the_two_sides_outranks_a_single_strong_opinion():
    # The property RRF exists for: a chunk both retrievers placed modestly
    # beats one that only a single retriever loved. Neither retriever can
    # carry a result on its own.
    fused = fuse(
        [_dense_hit("agreed", 3), _dense_hit("dense-only", 1)],
        [_lexical("agreed", 3)],
        2,
    )
    assert [hit.id for hit in fused] == ["agreed", "dense-only"]


def test_a_chunk_buried_by_dense_is_rescued_by_a_strong_lexical_rank():
    # The Annex III shape, as arithmetic: the real corpus put a gold chunk
    # below dense rank 50 while BM25 ranked it first, and the fused result has
    # to pull it above a chunk dense merely liked. Stated at fuse() level
    # because a seven-chunk fixture has no rank 40 to be buried at, and
    # rescue-at-depth is the property, not the fixture's ordering.
    fused = fuse(
        [_dense_hit("buried", 40), _dense_hit("comfortable", 4)],
        [_lexical("buried", 1)],
        2,
    )
    assert [hit.id for hit in fused] == ["buried", "comfortable"]
    assert fused[0].dense_rank == 40 and fused[0].bm25_rank == 1


def test_a_chunk_one_side_ranked_hopelessly_does_not_drag_the_other_down():
    # The converse, and the reason fusion is on rank rather than score: BM25
    # put a gold chunk at 724 of 901 on the real corpus. A retriever that
    # wrong must be unable to overrule one that is right.
    fused = fuse(
        [_dense_hit("correct", 1)],
        [_lexical("correct", 724), _lexical("noise", 1)],
        2,
    )
    assert [hit.id for hit in fused] == ["correct", "noise"]


def test_fused_ranks_are_renumbered_one_based_and_sequential():
    fused = fuse([_dense_hit("a", 1), _dense_hit("b", 2)], [_lexical("c", 1)], 3)
    assert [hit.rank for hit in fused] == [1, 2, 3]


def test_fusion_truncates_to_top_k():
    fused = fuse([_dense_hit("a", 1), _dense_hit("b", 2)], [_lexical("c", 1)], 2)
    assert len(fused) == 2


def test_a_tie_is_broken_deterministically():
    # A dense-only chunk at rank 1 and a BM25-only chunk at rank 1 score
    # identically. Without a rule the order is dict insertion order, which
    # makes two eval runs over one corpus disagree and every delta unreadable.
    first = fuse([_dense_hit("zzz", 1)], [_lexical("aaa", 1)], 2)
    second = fuse([_dense_hit("zzz", 1)], [_lexical("aaa", 1)], 2)
    assert [h.id for h in first] == [h.id for h in second] == ["zzz", "aaa"]


# ---------------------------------------------------------------------------
# hybrid_search -- the two halves wired together
# ---------------------------------------------------------------------------


def test_hybrid_still_answers_a_question_the_dense_side_handles_alone(
    collection, bm25_index, fake_embedder
):
    hits = hybrid_search(
        collection,
        bm25_index,
        fake_embedder,
        "which AI practices are prohibited",
        3,
        10,
    )
    assert hits[0].id == "art_5.para_1"


def test_fusion_preserves_the_dense_rank_and_cosine_it_was_given(
    collection, bm25_index, fake_embedder
):
    # fuse() overwrites rank and score with the fused values, and must leave
    # the dense provenance alone -- it is what tells an eval miss apart from a
    # ranking problem. Cross-checked against search() rather than against a
    # constant, so it states "fusion changed nothing here" and not a cosine.
    question = "which AI practices are prohibited"
    dense = {hit.id: hit for hit in search(collection, fake_embedder, question, 7)}
    hits = hybrid_search(collection, bm25_index, fake_embedder, question, 5, 10)
    for hit in hits:
        assert hit.dense_rank == dense[hit.id].rank
        assert hit.dense_score == dense[hit.id].score
        # And the fused score really is a different number from the cosine.
        assert hit.score != hit.dense_score


def test_hybrid_returns_nothing_for_a_top_k_of_zero(
    collection, bm25_index, fake_embedder
):
    assert hybrid_search(collection, bm25_index, fake_embedder, "risk", 0, 10) == []


def test_hybrid_over_an_empty_collection_still_returns_lexical_hits(
    empty_collection, bm25_index, fake_embedder
):
    # An empty vector store is what a half-built index looks like. Retrieval
    # degrading to lexical-only beats returning nothing, and either way it
    # must not raise.
    hits = hybrid_search(
        empty_collection, bm25_index, fake_embedder, "regulatory sandbox", 3, 10
    )
    assert [hit.id for hit in hits] == ["art_57.para_1"]
    assert hits[0].dense_rank is None and hits[0].dense_score is None


def test_a_lexical_only_hit_carries_the_citation_the_answer_will_quote(
    empty_collection, bm25_index, fake_embedder
):
    # A lexical-only hit is built from the chunk record rather than from
    # Chroma, so it is the one that could silently arrive without a citation --
    # and the generator refuses anything it cannot cite. Driven through an
    # empty vector store so the dense path cannot supply the fields by luck.
    hits = hybrid_search(
        empty_collection, bm25_index, fake_embedder, "regulatory sandbox", 1, 10
    )
    assert hits[0].article_no == "Article 57(1)"
    assert hits[0].title == "AI regulatory sandboxes"
    assert hits[0].kind == "article"
    assert hits[0].source_url.startswith("https://")
    assert "regulatory sandbox" in hits[0].text


def test_hybrid_ranks_are_one_based_and_sequential(
    collection, bm25_index, fake_embedder
):
    hits = hybrid_search(collection, bm25_index, fake_embedder, "high risk", 5, 10)
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# The log gains the component ranks
# ---------------------------------------------------------------------------


def test_the_log_records_which_retriever_found_each_chunk(
    tmp_path, empty_collection, bm25_index, fake_embedder
):
    # Through an empty vector store, so the null in dense_ranks is a fact
    # about provenance rather than an accident of the fixture's ordering.
    hits = hybrid_search(
        empty_collection, bm25_index, fake_embedder, "regulatory sandbox", 2, 10
    )
    log_query("regulatory sandbox", hits, tmp_path)
    entry = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8"))
    assert entry["bm25_ranks"][0] == 1
    assert entry["dense_ranks"][0] is None
    assert len(entry["dense_ranks"]) == len(entry["chunk_ids"])
