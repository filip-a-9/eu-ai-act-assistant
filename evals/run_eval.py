"""Score retrieval against evals/questions.yaml.

This is the other test surface. pytest asserts deterministic facts about
structure and contract and never touches the network; this calls the real API
against the real index and reports a number that moves. Neither replaces the
other: green tests say the plumbing is connected, this says whether the
plumbing is any good.

No threshold is enforced and nothing here exits non-zero for a low score. A
quality bar encoded as a build failure is a bar people learn to route around.
The obligation is to report the number, including when it got worse.

CLAUDE.md: touched retrieval, re-run this and report recall@5 before and after.

Usage:
    python evals/run_eval.py              # recall at the configured top_k
    python evals/run_eval.py -k 10        # at a different k
    python evals/run_eval.py --sweep      # at 1, 3, 5, 10 and 20
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.embed import openai_embedder
from core.index import open_collection
from core.retrieve import Hit, search

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
SWEEP = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expect: list[str]
    article_no: str
    group: str  # "statutory" (the Act's own vocabulary) or "lay"


@dataclass(frozen=True)
class Result:
    question: Question
    hits: list[Hit]

    @property
    def gold_rank(self) -> int | None:
        """1-based rank of the first expected chunk, or None if absent."""
        for hit in self.hits:
            if hit.id in self.question.expect:
                return hit.rank
        return None

    def recall_at(self, k: int) -> bool:
        rank = self.gold_rank
        return rank is not None and rank <= k


def load_questions() -> list[Question]:
    raw = yaml.safe_load(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = [
        Question(
            id=entry["id"],
            question=entry["question"],
            expect=list(entry["expect"]),
            article_no=entry.get("article_no", ""),
            group=entry.get("group", "statutory"),
        )
        for entry in raw
    ]
    duplicates = {q.id for q in questions if [x.id for x in questions].count(q.id) > 1}
    if duplicates:
        raise SystemExit(
            f"duplicate question ids in questions.yaml: {sorted(duplicates)}"
        )
    return questions


def check_gold_exists(questions: list[Question], collection) -> None:
    """Fail before spending anything if a gold id is not in the index.

    A typo in `expect` is indistinguishable from a retrieval failure once the
    numbers are printed -- it just looks like a miss. Catching it here is the
    difference between fixing a typo and chasing a ranking ghost.
    """
    wanted = sorted({chunk_id for q in questions for chunk_id in q.expect})
    found = set(collection.get(ids=wanted, include=[])["ids"])
    missing = [chunk_id for chunk_id in wanted if chunk_id not in found]
    if missing:
        raise SystemExit(
            "these expected chunk ids are not in the index: "
            + ", ".join(missing)
            + "\nEither the gold is a typo or the chunker's granularity changed."
        )


def run(questions: list[Question], collection, embedder, top_k: int) -> list[Result]:
    results = []
    for question in questions:
        hits = search(collection, embedder, question.question, top_k)
        results.append(Result(question=question, hits=hits))
    return results


def recall(results: list[Result], k: int) -> float:
    return sum(result.recall_at(k) for result in results) / len(results)


def mrr(results: list[Result], k: int) -> float:
    """Mean reciprocal rank, counting only gold found at or above ``k``.

    Cut off at k like recall is. Without the cut every row of a sweep would
    print the same number, since the rank of the first gold chunk does not
    depend on how many results were displayed.
    """
    total = 0.0
    for result in results:
        rank = result.gold_rank
        if rank is not None and rank <= k:
            total += 1.0 / rank
    return total / len(results)


def report(results: list[Result], top_k: int) -> None:
    print(f"questions  {len(results)}")
    print(f"retrieved  top {top_k}")
    print()
    for k in sorted({1, 3, top_k}):
        print(f"recall@{k:<4} {recall(results, k):.2f}")
    print(f"MRR@{top_k:<7} {mrr(results, top_k):.3f}")

    # Split by how the question is phrased. The headline number averages two
    # very different populations: questions using the Act's own vocabulary
    # match on terminology almost for free, because every chunk is embedded
    # with its citation header. Whether a retrieval change helped is almost
    # entirely a question about the lay group.
    groups = sorted({result.question.group for result in results})
    if len(groups) > 1:
        print(f"\ngroup        n   recall@{top_k}  MRR@{top_k}")
        for group in groups:
            subset = [r for r in results if r.question.group == group]
            print(
                f"{group:<12} {len(subset):<3} {recall(subset, top_k):.2f}"
                f"      {mrr(subset, top_k):.3f}"
            )

    misses = [result for result in results if not result.recall_at(top_k)]
    if not misses:
        print("\nno misses")
        return

    # Misses are printed with what came back instead, because an aggregate
    # number tells you retrieval got worse and nothing about where to look.
    print(f"\n{len(misses)} miss{'es' if len(misses) != 1 else ''}:")
    for result in misses:
        print(f"\n  {result.question.id}")
        print(f"    asked     {result.question.question}")
        print(f"    wanted    {', '.join(result.question.expect)}")
        got = ", ".join(f"{hit.id} ({hit.score:.2f})" for hit in result.hits[:5])
        print(f"    got       {got or '(nothing)'}")

    # A gold chunk sitting just outside the cut is a different problem from one
    # retrieval cannot find at all: the first is a k or a re-ranking question,
    # the second is an embedding question.
    deep = [r for r in misses if r.gold_rank is not None]
    if deep:
        print(
            f"\n  {len(deep)} of those did retrieve a gold chunk, below the cut: "
            + ", ".join(f"{r.question.id}@{r.gold_rank}" for r in deep)
        )


def sweep(results_by_k: dict[int, list[Result]]) -> None:
    print("\nk    recall   MRR")
    for k, results in sorted(results_by_k.items()):
        print(f"{k:<5}{recall(results, k):.2f}     {mrr(results, k):.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score retrieval over questions.yaml.")
    parser.add_argument(
        "-k", "--top-k", type=int, default=None,
        help="how many chunks to retrieve (default: TOP_K from the environment)",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help=f"also report recall and MRR at k = {', '.join(map(str, SWEEP))}",
    )
    args = parser.parse_args()

    config = load_config()
    top_k = args.top_k if args.top_k is not None else config.top_k

    questions = load_questions()
    collection = open_collection(config.index_dir)
    check_gold_exists(questions, collection)
    embedder = openai_embedder(config.openai_api_key, config.embed_model)

    results = run(questions, collection, embedder, top_k)
    report(results, top_k)

    if args.sweep:
        # Retrieved once at the largest k and truncated, rather than querying
        # five times: the ranking is identical and it is four fewer API calls
        # per question.
        widest = run(questions, collection, embedder, max(SWEEP))
        sweep({k: widest for k in SWEEP})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
