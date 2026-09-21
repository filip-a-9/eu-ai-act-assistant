"""Ask the AI Act a question at the terminal, and keep asking.

The conversational half of the assistant. ``scripts/query.py`` is deliberately
kept generation-free next to this one: when an answer here is wrong, running
the same question there says whether retrieval or generation is responsible.

It is a loop rather than a one-shot command because the rewrite node only does
anything on a follow-up. "Which practices are banned?" then "what about small
companies?" is the path that exercises it, and a flag-driven history nobody
would type by hand would leave that path unexercised until the UI existed.

Argument parsing and formatting only: every decision is in core/.

Usage:
    python scripts/ask.py
    python scripts/ask.py -k 8
    python scripts/ask.py "which AI practices are banned?"   # one turn, then exit
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.embed import openai_embedder
from core.generate import Answer, openai_generator
from core.graph import build_graph, run_turn
from core.index import open_collection

WIDTH = 88

# An explicit mid-gray. ANSI dim (\033[2m) renders as black on a black
# terminal, which is how metadata becomes invisible on this machine.
GRAY = "\033[38;5;245m"
RESET = "\033[0m"


def gray(text: str) -> str:
    return f"{GRAY}{text}{RESET}" if sys.stdout.isatty() else text


def wrap(text: str) -> list[str]:
    """Wrap to the terminal width without flattening the model's own layout.

    ``textwrap.wrap`` over the whole answer collapses every newline, which
    turns a list of ten prohibitions into one unreadable paragraph.
    """
    lines: list[str] = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line, width=WIDTH) or [""])
    return lines


def render(answer: Answer) -> str:
    """The answer, then the provisions it actually cited."""
    lines = wrap(answer.text)
    if answer.cited_hits:
        lines.append("")
        for hit in answer.cited_hits:
            label = hit.article_no
            if hit.title and hit.title != label:
                label += f" — {hit.title}"
            lines.append(gray(f"  {label}"))
            lines.append(gray(f"  {hit.source_url}"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the EU AI Act a question, with citations or a refusal."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="ask one question and exit; omit it for a conversation",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="how many chunks to retrieve (default: TOP_K from the environment)",
    )
    args = parser.parse_args()

    config = load_config()
    graph = build_graph(
        collection=open_collection(config.index_dir, config.scratch_dir),
        embedder=openai_embedder(config.openai_api_key, config.embed_model),
        generator=openai_generator(
            config.openai_api_key, config.chat_model, config.temperature
        ),
        top_k=args.top_k if args.top_k is not None else config.top_k,
        logs_dir=config.logs_dir,
    )

    if args.question:
        print(render(run_turn(graph, args.question, history=[])))
        return 0

    print(gray(f"EU AI Act · {config.chat_model} · blank line or Ctrl-D to leave"))
    # History lives here rather than in the graph: one turn is a pure function
    # of the question and what came before it, which is what makes a turn
    # reproducible from the log.
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            return 0

        answer = run_turn(graph, question, history=history)
        print()
        print(render(answer))
        history.append((question, answer.text))


if __name__ == "__main__":
    raise SystemExit(main())
