"""Ask the index a question and print the ranked chunks it returns.

This is the retrieval half of the assistant with nothing generated on top: no
model writes an answer here, so what you read is exactly what the generator
will be given. That makes it the right place to judge whether a bad answer is
a retrieval problem or a generation problem.

Argument parsing and formatting only. Every ranking decision lives in
core/retrieve.py, so that the CLI, the eval and the app cannot drift apart.

Usage:
    python scripts/query.py "what AI practices are prohibited?"
    python scripts/query.py -k 10 "obligations for providers of GPAI models"
    python scripts/query.py --full "when does the Act apply to research?"
    python scripts/query.py --json "penalties" | python -m json.tool
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.embed import openai_embedder
from core.index import open_collection
from core.retrieve import Hit, hit_to_dict, log_query, search

SNIPPET_CHARS = 300


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The opening of a chunk, cut at a word boundary."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return textwrap.shorten(collapsed, width=limit, placeholder=" ...")


def render(hit: Hit, full: bool) -> str:
    body = hit.text if full else snippet(hit.text)
    heading = f"{hit.rank:>2}  {hit.score:.3f}  {hit.article_no}"
    if hit.title and hit.title != hit.article_no:
        heading += f" — {hit.title}"

    lines = [heading]
    if hit.chapter:
        lines.append(f"            {hit.chapter}")
    lines.extend(
        textwrap.wrap(
            body, width=92, initial_indent=" " * 12, subsequent_indent=" " * 12
        )
    )
    lines.append(f"            {hit.source_url}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve the chunks of the EU AI Act nearest a question."
    )
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="how many chunks to return (default: TOP_K from the environment)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print each chunk in full instead of a snippet",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the hits as JSON, for piping",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="skip the per-query log entry",
    )
    args = parser.parse_args()

    config = load_config()
    top_k = args.top_k if args.top_k is not None else config.top_k

    collection = open_collection(config.index_dir)
    embedder = openai_embedder(config.openai_api_key, config.embed_model)
    hits = search(collection, embedder, args.question, top_k)

    if not args.no_log:
        log_query(args.question, hits, config.logs_dir)

    if args.json:
        print(
            json.dumps([hit_to_dict(hit) for hit in hits], ensure_ascii=False, indent=2)
        )
        return 0

    if not hits:
        # Not an error. Nothing supporting the question is the case that must
        # end in a refusal rather than a guess, so it prints plainly.
        print(f'no chunk matched "{args.question}"')
        return 0

    print(f'"{args.question}"  ·  {len(hits)} of {collection.count():,} chunks\n')
    for hit in hits:
        print(render(hit, args.full))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
