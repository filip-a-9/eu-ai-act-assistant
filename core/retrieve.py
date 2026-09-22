"""Find the chunks that might answer a question.

Two retrievers, fused. ``search`` is dense: it compares the question's
embedding against the chunks'. ``Bm25Index`` is lexical: it counts words,
weighting rare ones heavily and long chunks down. Legal text needs both --
embeddings blur exact tokens like "Annex III" into a generic annex-shaped
direction, and a word counter is helpless against a question that avoids the
Act's vocabulary entirely.

Measured before this was built, over the real corpus and the eight
lay-phrased eval questions, BM25 alone put a gold chunk in the top five once;
dense managed it three times. Neither is reliably better. What BM25 adds is
the one case it owns outright: Annex III point 4 contains "analyse and filter
job applications" verbatim, so lexical matching ranks first what dense could
not find at all.

That is what picks the fusion rule. Either retriever may be catastrophically
wrong on any given question -- rank 724 out of 901, not a near miss -- so
fusion happens on *rank*, via Reciprocal Rank Fusion, never on score. A chunk
one side ranked 724th contributes ~0 instead of poisoning the ranking. It also
sidesteps a weight, which DECISIONS.md refuses to pick on taste, and a score
normalisation between a bounded cosine and an unbounded BM25 score.

Nothing in this module interprets the text it retrieves. A chunk is data.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from core.embed import Embedder

LOG_FILENAME = "queries.jsonl"

# The RRF constant, from Cormack et al. (2009), which found 60 insensitive
# enough across collections to be worth fixing. It flattens the curve so that
# ranks 1 and 3 are not wildly far apart, which is exactly the property that
# stops one confident retriever from overruling agreement between both. Not in
# core/config.py on purpose: it is a property of the algorithm, not a dial a
# deployment trades latency against recall on. That dial is fusion_candidates.
RRF_K = 60

# Lowercase alphanumeric runs. Shared by the corpus and the query so the two
# cannot drift -- a tokeniser that split them differently would silently match
# nothing, which looks exactly like a corpus with no relevant chunk in it.
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, ranked and scored.

    Frozen because a Hit travels from retrieval through generation to the UI;
    a stage that adjusted a score in passing would be very hard to find.
    """

    rank: int  # 1-based, best first
    id: str
    # After fusion this is the RRF value, roughly 0.016 to 0.033. It orders the
    # list and means nothing on its own -- do not render it to a reader, and do
    # not compare it across queries. ``dense_score`` below is the readable one.
    score: float
    article_no: str  # the citation label, never empty
    title: str
    chapter: str
    kind: str  # article | annex | recital
    source_url: str
    text: str  # the law, clean -- no citation header welded on

    # Where each retriever placed this chunk, and None where that retriever
    # never saw it. These are what diagnose a miss: a gold chunk at dense 400
    # and bm25 1 is a different problem from one both sides buried, and the
    # fused score alone cannot tell the two apart.
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None  # cosine similarity in [0, 1]


def search(
    collection: Any,
    embedder: Embedder,
    question: str,
    top_k: int,
) -> list[Hit]:
    """Return the ``top_k`` chunks nearest the question, best first.

    Chroma reports cosine *distance*, where 0 is a perfect match. Every
    surface above this one wants a similarity, so the conversion happens once,
    here, rather than in each caller getting the direction right by luck.
    """
    if top_k <= 0:
        return []

    result = collection.query(
        query_embeddings=embedder.embed([question]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma nests one list per query embedding; we only ever send one.
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        Hit(
            rank=rank,
            id=chunk_id,
            score=1.0 - distance,
            article_no=metadata["article_no"],
            title=metadata["title"],
            chapter=metadata["chapter"],
            kind=metadata["kind"],
            source_url=metadata["source_url"],
            text=document,
            # A dense hit knows it came from dense, at this rank and this
            # cosine. fuse() then overwrites rank and score with the fused
            # values and leaves these two alone, so the provenance survives.
            dense_rank=rank,
            dense_score=1.0 - distance,
        )
        # strict: these four lists are one response destructured into columns,
        # not four independent sequences. Chroma builds them the same length,
        # so a mismatch means the response is malformed -- and zip's default
        # would answer that by returning fewer hits than were retrieved, which
        # in a citation-grounded assistant is a citation dropped in silence.
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True), start=1
        )
    ]


class Bm25Index:
    """BM25 over the chunk corpus, held in memory.

    Built from ``data/chunks/chunks.jsonl`` at open time rather than persisted
    beside the vector index: measured at 40ms for 901 chunks with no API call,
    and a second committed artifact is a second thing that can go stale against
    the vectors. ``rank_bm25`` recomputes its IDF table on construction anyway,
    so persisting would save about half of that 40ms and buy a sync problem.

    Indexes ``embed_text``, not ``text``. The citation header is where
    "Annex III, point 1" exists as literal tokens, and that is the exact-match
    case this retriever is here to serve. It is also what the vector store
    embedded, so both sides read the same string.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = list(records)
        corpus = [_tokenise(record["embed_text"]) for record in self._records]
        # BM25Okapi divides by the average document length, so an empty corpus
        # raises ZeroDivisionError on construction. A corpus can legitimately
        # be empty on a fresh clone before build_chunks.py has run, and that
        # should degrade to dense-only retrieval rather than crash.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> Bm25Index:
        return cls(records)

    def search(self, question: str, n: int) -> list[tuple[dict[str, Any], int]]:
        """Return up to ``n`` ``(record, rank)`` pairs, best first.

        Chunks scoring zero are dropped rather than padded out to ``n``. A zero
        means the chunk shares no term with the question, and handing it to
        fusion would give it credit for appearing in a list it had no opinion
        about -- which is how a lexical retriever poisons a ranking it should
        have abstained from.
        """
        if n <= 0 or self._bm25 is None:
            return []
        tokens = _tokenise(question)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # strict: one score per record by construction, and a mismatch would
        # mean scores had silently been computed against a different corpus.
        ranked = sorted(
            (
                (score, record)
                for score, record in zip(scores, self._records, strict=True)
                if score > 0.0
            ),
            # Ties broken on id so two runs over one corpus never disagree.
            key=lambda pair: (-pair[0], pair[1]["id"]),
        )
        return [(record, rank) for rank, (_, record) in enumerate(ranked[:n], start=1)]


def _hit_from_record(
    record: dict[str, Any], rank: int, score: float, bm25_rank: int
) -> Hit:
    """Build a Hit for a chunk only the lexical side found.

    The dense path gets its fields back from Chroma; this one reads them off
    the chunk record instead. Both must produce a Hit carrying a citation,
    because the generator refuses anything it cannot cite.
    """
    return Hit(
        rank=rank,
        id=record["id"],
        score=score,
        article_no=record["article_no"],
        title=record["title"],
        chapter=record["chapter"],
        kind=record["kind"],
        source_url=record["source_url"],
        text=record["text"],
        dense_rank=None,
        bm25_rank=bm25_rank,
        dense_score=None,
    )


def fuse(
    dense: list[Hit],
    lexical: list[tuple[dict[str, Any], int]],
    top_k: int,
) -> list[Hit]:
    """Combine two rankings by Reciprocal Rank Fusion, best first.

    Each retriever contributes ``1 / (RRF_K + rank)`` for every chunk it
    returned, and the sums are sorted. A chunk both sides placed modestly beats
    one a single side loved, which is the whole point: neither retriever can
    carry a result alone.
    """
    if top_k <= 0:
        return []

    dense_by_id = {hit.id: hit for hit in dense}
    lexical_by_id = {record["id"]: (record, rank) for record, rank in lexical}

    scores: dict[str, float] = {}
    for hit in dense:
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + hit.rank)
    for record, rank in lexical:
        chunk_id = record["id"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

    # A dense-only chunk at rank 1 and a lexical-only chunk at rank 1 score
    # identically, and ties here are common rather than exotic. Ordering them
    # by dict insertion would make two runs over one corpus disagree, and an
    # eval delta that moves for that reason is unreadable. Dense wins a tie:
    # it is the retriever with the recorded recall number behind it.
    order = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            chunk_id not in dense_by_id,
            chunk_id,
        ),
    )

    fused: list[Hit] = []
    for rank, chunk_id in enumerate(order[:top_k], start=1):
        entry = lexical_by_id.get(chunk_id)
        bm25_rank = entry[1] if entry is not None else None
        dense_hit = dense_by_id.get(chunk_id)
        if dense_hit is not None:
            fused.append(
                replace(
                    dense_hit,
                    rank=rank,
                    score=scores[chunk_id],
                    bm25_rank=bm25_rank,
                )
            )
        else:
            # Unreachable unless the chunk came from one side or the other.
            assert entry is not None
            fused.append(_hit_from_record(entry[0], rank, scores[chunk_id], entry[1]))
    return fused


def hybrid_search(
    collection: Any,
    bm25: Bm25Index,
    embedder: Embedder,
    question: str,
    top_k: int,
    candidates: int,
) -> list[Hit]:
    """Retrieve from both sides and fuse. The entry point every caller uses.

    ``candidates`` is how deep each retriever reaches before fusion. It is
    larger than ``top_k`` on purpose: a chunk the vector store ranked 40th is
    only rescuable if it is still in the list when the rankings meet.
    """
    if top_k <= 0:
        return []
    dense = search(collection, embedder, question, candidates)
    lexical = bm25.search(question, candidates)
    return fuse(dense, lexical, top_k)


def log_query(
    question: str,
    hits: list[Hit],
    logs_dir: Path,
    rewritten: str | None = None,
) -> None:
    """Append one JSON line recording what was asked and what came back.

    CLAUDE.md requires question, rewritten question, chunk ids and scores on
    every query. ``rewritten`` is written as null until the graph adds a
    rewrite node: present-and-null keeps the log format stable, where an
    absent key would mean every existing line needed special-casing later.

    The chunk *text* is deliberately not logged. The ids identify it exactly,
    and a log that duplicates the corpus is one that nobody will read.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": question,
        "rewritten": rewritten,
        "chunk_ids": [hit.id for hit in hits],
        "scores": [hit.score for hit in hits],
        # Which retriever found each chunk, and where. Additive: the keys
        # CLAUDE.md requires are all still above, and lines written before
        # hybrid retrieval stay parseable without a migration. Nulls are
        # meaningful here -- they say a retriever never returned this chunk,
        # which is the difference between a ranking problem and a blind spot.
        "dense_ranks": [hit.dense_rank for hit in hits],
        "bm25_ranks": [hit.bm25_rank for hit in hits],
    }
    with (logs_dir / LOG_FILENAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """A Hit as plain data, for ``--json`` output and the eval report."""
    return asdict(hit)
