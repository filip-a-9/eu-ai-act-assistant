"""Parse data/raw/ into data/chunks/chunks.jsonl.

Offline step, run by hand, no network and no OpenAI key. Kept separate from
fetching for the same reason fetching is separate from indexing: the steps have
different failure modes and different reasons to re-run.

The output is committed to the repository. That is deliberate -- a change to the
chunker then shows up as a readable diff over the corpus rather than as an
invisible change in an artefact nobody can see.

Usage:
    python scripts/build_chunks.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.chunker import (
    Chunk,
    parse_consolidated,
    parse_oj_recitals,
    unrecognised_labels,
)
from core.config import chunks_dir, raw_dir

# Asserted so that a future re-fetch which changes EUR-Lex's markup fails here,
# loudly, instead of silently producing a smaller and worse corpus. Changing
# the chunker legitimately changes this number -- edit it in the same commit.
EXPECTED_CHUNKS = 901

# Articles 105-107, 109 and 110 amend other instruments by quoting the text
# they insert, so their markers carry the *other* instrument's numbering
# ("‘5."). Those articles stay whole. Any marker outside this set is a drafting
# form the chunker has not seen, and folding it away silently would cost every
# subdivision in that article.
KNOWN_QUOTED_INSERTIONS = [
    ("art_105", "‘5"),
    ("art_106", "‘12"),
    ("art_107", "‘4"),
    ("art_109", "‘3"),
    ("art_110", "‘(68"),
]

# The embedding model's ceiling is 8,191 tokens. Roughly four characters per
# token leaves the largest chunk (Article 5(1), 5,406 characters, since Annex
# III now splits at its areas) with ample headroom, but the check is cheap and
# a silent truncation at index time would not be.
EMBED_CHAR_CEILING = 8191 * 4


def load_sources() -> tuple[str, str]:
    """Read both documents by the filenames the manifest recorded."""
    source = raw_dir()
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path}. Run: python scripts/fetch_source.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        name: source / manifest[name]["filename"]
        for name in ("consolidated", "oj_as_published")
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(
                f"missing {path} for '{name}'. Run: python scripts/fetch_source.py"
            )
    return (
        paths["consolidated"].read_text(encoding="utf-8"),
        paths["oj_as_published"].read_text(encoding="utf-8"),
    )


def check(chunks: list[Chunk], consolidated_html: str) -> list[str]:
    """Return every problem found, rather than raising on the first."""
    problems = []

    folded = sorted(unrecognised_labels(consolidated_html))
    if folded != sorted(KNOWN_QUOTED_INSERTIONS):
        unexpected = [f for f in folded if f not in KNOWN_QUOTED_INSERTIONS]
        gone = [f for f in KNOWN_QUOTED_INSERTIONS if f not in folded]
        problems.append(
            f"subdivision markers changed -- unexpected: {unexpected}, "
            f"no longer present: {gone}"
        )

    if len(chunks) != EXPECTED_CHUNKS:
        problems.append(
            f"expected {EXPECTED_CHUNKS} chunks, produced {len(chunks)} -- "
            "the source markup or the chunker changed"
        )

    duplicates = [i for i, n in Counter(c.id for c in chunks).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate ids: {duplicates[:5]}")

    empty = [c.id for c in chunks if not c.text.strip()]
    if empty:
        problems.append(f"{len(empty)} chunks with empty text: {empty[:5]}")

    uncited = [c.id for c in chunks if not c.article_no.strip()]
    if uncited:
        problems.append(f"{len(uncited)} chunks with no citation: {uncited[:5]}")

    marked = [c.id for c in chunks if "▼" in c.text or "►" in c.text]
    if marked:
        problems.append(f"{len(marked)} chunks retain amendment markers: {marked[:5]}")

    oversized = [c.id for c in chunks if len(c.embed_text) > EMBED_CHAR_CEILING]
    if oversized:
        problems.append(
            f"{len(oversized)} chunks exceed the embedding ceiling: {oversized[:5]}"
        )

    return problems


def summarise(chunks: list[Chunk]) -> None:
    by_kind = Counter(c.kind for c in chunks)
    lengths = sorted(len(c.text) for c in chunks)
    largest = max(chunks, key=lambda c: len(c.text))
    print(f"chunks     {len(chunks):,}")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:<8} {count:,}")
    print(
        f"text chars min {lengths[0]:,}  median {lengths[len(lengths) // 2]:,}  "
        f"max {lengths[-1]:,} ({largest.article_no})"
    )


def main() -> int:
    consolidated_html, oj_html = load_sources()
    chunks = parse_consolidated(consolidated_html) + parse_oj_recitals(oj_html)

    problems = check(chunks, consolidated_html)
    summarise(chunks)
    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    destination = chunks_dir()
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "chunks.jsonl"
    # newline="\n" so the committed corpus is byte-identical on Windows and
    # Linux; .gitattributes normalises it, but writing LF avoids the whole file
    # showing as modified on every build.
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                **chunk.metadata(),
                "text": chunk.text,
                "embed_text": chunk.embed_text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
