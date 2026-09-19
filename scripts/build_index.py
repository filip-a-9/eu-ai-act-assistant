"""Embed data/chunks/chunks.jsonl into the persisted Chroma index.

Offline step, run by hand. This is the only script that spends money, and the
only one that needs OPENAI_API_KEY. The app never runs it: CLAUDE.md requires
the app to open a prebuilt index read-only, and the index is committed to the
repository so that a host with no build step still has one.

Whole-corpus facts are asserted here rather than in pytest, for the same
reason build_chunks.py asserts its count here: the test suite must run on a
fresh clone with no corpus and no API key.

Usage:
    python scripts/build_index.py            # rebuild from data/chunks
    python scripts/build_index.py --dry-run  # cost and plan only, no API calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.embed import openai_embedder  # noqa: E402
from core.index import (  # noqa: E402
    COLLECTION_NAME,
    MANIFEST_FILENAME,
    build,
    load_chunk_records,
    open_collection,
)
from core.config import load_config  # noqa: E402

# text-embedding-3-small, USD per million tokens, as of 2026-09. Used only to
# print an estimate before spending anything -- nothing depends on it being
# exactly right, and it is here so the cost of a rebuild is never a surprise.
USD_PER_MILLION_TOKENS = 0.02

# The tokeniser is not a dependency of this project and is not worth adding to
# print an estimate. Four characters per token is the usual rule of thumb for
# English prose and runs slightly conservative on legal text.
CHARS_PER_TOKEN = 4


def estimate(records: list[dict]) -> tuple[int, float]:
    characters = sum(len(record["embed_text"]) for record in records)
    tokens = characters // CHARS_PER_TOKEN
    return tokens, tokens / 1_000_000 * USD_PER_MILLION_TOKENS


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def summarise(records: list[dict]) -> None:
    by_kind = Counter(record["kind"] for record in records)
    tokens, cost = estimate(records)
    print(f"chunks     {len(records):,}")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:<8} {count:,}")
    print(f"to embed   ~{tokens:,} tokens  (~${cost:.4f})")


def write_manifest(
    index_dir: Path, chunks_path: Path, records: list[dict], model: str, count: int
) -> Path:
    """Record what this index was built from.

    The index is committed, so "are these vectors current for these chunks?"
    is otherwise unanswerable by looking at the repository -- a binary blob
    reveals nothing in a diff. The chunks digest is what makes a stale index
    detectable.
    """
    manifest = {
        "collection": COLLECTION_NAME,
        "embed_model": model,
        "chunk_count": count,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunks_file": chunks_path.name,
        "chunks_sha256": sha256_of(chunks_path),
    }
    path = index_dir / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the cost estimate without calling the API",
    )
    args = parser.parse_args()

    config = load_config()
    records = load_chunk_records(config.chunks_dir)
    if not records:
        print("no chunks to index. Run: python scripts/build_chunks.py")
        return 1

    summarise(records)
    if args.dry_run:
        print("\ndry run -- nothing embedded, nothing written")
        return 0

    print(f"\nembedding with {config.embed_model}...")
    embedder = openai_embedder(config.openai_api_key, config.embed_model)
    count = build(
        records,
        embedder,
        config.index_dir,
        progress=lambda done, total: print(f"  {done:,}/{total:,}", flush=True),
    )

    problems = []
    if count != len(records):
        problems.append(
            f"indexed {count} of {len(records)} chunks -- the write was incomplete"
        )

    # Reopened rather than trusting the handle that just wrote it: this is the
    # path the app will take, and it is the one worth proving works.
    reopened = open_collection(config.index_dir)
    if reopened.count() != len(records):
        problems.append(
            f"reopened index holds {reopened.count()} chunks, expected {len(records)}"
        )

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    chunks_path = config.chunks_dir / "chunks.jsonl"
    manifest_path = write_manifest(
        config.index_dir, chunks_path, records, config.embed_model, count
    )

    size_mb = directory_bytes(config.index_dir) / 1_048_576
    print(f"\nindexed    {count:,} chunks into {config.index_dir}")
    print(f"on disk    {size_mb:.1f} MB (committed to the repository)")
    print(f"manifest   {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
