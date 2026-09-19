"""Find the chunks that might answer a question.

Dense retrieval only, for now. The stack names ``rank_bm25`` and legal text is
full of exact tokens that embeddings blur -- "Annex III", "general-purpose AI
model" -- so hybrid search is coming. It is deliberately not here yet: fusing
two rankings needs a weight, and a weight chosen before there is a recall
number to move is chosen by taste. ``search`` returns a ranked list of scored
candidates, which is the shape a fusion step consumes, so adding BM25 later
adds a function rather than rewriting this one.

Nothing in this module interprets the text it retrieves. A chunk is data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.embed import Embedder

LOG_FILENAME = "queries.jsonl"


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk, ranked and scored.

    Frozen because a Hit travels from retrieval through generation to the UI;
    a stage that adjusted a score in passing would be very hard to find.
    """

    rank: int          # 1-based, best first
    id: str
    score: float       # cosine similarity in [0, 1]; higher is better
    article_no: str    # the citation label, never empty
    title: str
    chapter: str
    kind: str          # article | annex | recital
    source_url: str
    text: str          # the law, clean -- no citation header welded on


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
        )
        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        )
    ]


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
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "rewritten": rewritten,
        "chunk_ids": [hit.id for hit in hits],
        "scores": [hit.score for hit in hits],
    }
    with (logs_dir / LOG_FILENAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def hit_to_dict(hit: Hit) -> dict[str, Any]:
    """A Hit as plain data, for ``--json`` output and the eval report."""
    return asdict(hit)
