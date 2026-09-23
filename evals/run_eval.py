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
    python evals/run_eval.py --followups  # the rewrite, over followups.yaml
    python evals/run_eval.py --floor      # choose MIN_DENSE_SCORE, offtopic.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import load_config
from core.embed import openai_embedder
from core.generate import Generator, openai_generator, rewrite
from core.index import open_bm25, open_collection
from core.retrieve import Bm25Index, Hit, hybrid_search

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
FOLLOWUPS_PATH = Path(__file__).parent / "followups.yaml"
OFFTOPIC_PATH = Path(__file__).parent / "offtopic.yaml"
# Candidate floors printed by --floor: fine steps where in-scope and unrelated
# questions meet (0.177 against 0.190 when 0.18 was chosen), coarse above it.
FLOORS = (0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SWEEP = (1, 3, 5, 10, 20)
# The three ways each follow-up is retrieved, in report order.
CONDITIONS = ("typed", "rewrite", "standalone")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expect: list[str]
    article_no: str
    # "statutory" (the Act's own vocabulary) or "lay"; under --followups, the
    # condition the item was retrieved in.
    group: str


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
        """True when *any* expected chunk is at or above rank ``k``."""
        rank = self.gold_rank
        return rank is not None and rank <= k

    def missing_at(self, k: int) -> list[str]:
        """Expected chunks absent from the top ``k``, in the order declared.

        This is the actionable output of the whole script. An aggregate says
        retrieval is worse than it looked; this says which provision is the
        one nobody is being shown.
        """
        found = {hit.id for hit in self.hits if hit.rank <= k}
        return [chunk_id for chunk_id in self.question.expect if chunk_id not in found]

    def covered_at(self, k: int) -> bool:
        """True when *every* expected chunk is at or above rank ``k``.

        The strict counterpart to ``recall_at``. A question expecting four
        chunks scores a full hit under ``recall_at`` on one of them, which is
        how Annex III stayed absent from the top fifty while the question that
        names it scored 1.00.
        """
        return not self.missing_at(k)


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


@dataclass(frozen=True)
class Followup:
    id: str
    history: list[tuple[str, str]]
    question: str
    standalone: str
    expect: list[str]


def load_followups() -> list[Followup]:
    raw = yaml.safe_load(FOLLOWUPS_PATH.read_text(encoding="utf-8"))
    items = [
        Followup(
            id=entry["id"],
            history=[(asked, replied) for asked, replied in entry["history"]],
            question=entry["question"].strip(),
            standalone=entry["standalone"].strip(),
            expect=list(entry["expect"]),
        )
        for entry in raw
    ]
    ids = [item.id for item in items]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise SystemExit(
            f"duplicate follow-up ids in followups.yaml: {sorted(duplicates)}"
        )
    return items


@dataclass
class CappedGenerator:
    """Passes calls through until ``limit`` is spent, then raises before calling.

    One instance per invocation, so the cap is a total and not a per-item
    allowance: a retry loop or a stray second call cannot quietly double the
    bill, it stops the run.
    """

    inner: Generator
    limit: int
    used: int = field(default=0)

    def complete(self, system: str, user: str) -> str:
        if self.used >= self.limit:
            raise RuntimeError(
                f"generator spend cap reached: {self.limit} calls already made"
            )
        self.used += 1
        return self.inner.complete(system, user)


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


def run(
    questions: list[Question],
    collection,
    bm25: Bm25Index,
    embedder,
    top_k: int,
    candidates: int,
) -> list[Result]:
    results = []
    for question in questions:
        hits = hybrid_search(
            collection, bm25, embedder, question.question, top_k, candidates
        )
        results.append(Result(question=question, hits=hits))
    return results


def recall(results: list[Result], k: int) -> float:
    """Fraction of questions with at least one expected chunk in the top k."""
    return sum(result.recall_at(k) for result in results) / len(results)


def recall_all(results: list[Result], k: int) -> float:
    """Fraction of questions with *every* expected chunk in the top k.

    Reported beside ``recall`` rather than replacing it: every earlier number
    in DECISIONS.md is quoted in the loose metric, and a measurement you can
    no longer compare against is a measurement you have lost.
    """
    return sum(result.covered_at(k) for result in results) / len(results)


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


def provenance(hit: Hit) -> str:
    """``art_5.para_1 [d1 b4]`` -- where each retriever placed this chunk.

    Printed instead of the fused score, which after RRF is about 0.03 and says
    nothing on its own. These two numbers do: a gold chunk at ``d400 b1`` was
    rescued by the lexical side, one at ``d3 b-`` means BM25 abstained, and one
    absent from both is an embedding problem rather than a ranking one.
    """
    dense = f"d{hit.dense_rank}" if hit.dense_rank is not None else "d-"
    lexical = f"b{hit.bm25_rank}" if hit.bm25_rank is not None else "b-"
    return f"{hit.id} [{dense} {lexical}]"


def report(results: list[Result], top_k: int) -> None:
    print(f"questions  {len(results)}")
    print(f"retrieved  top {top_k}")
    print()
    print("recall     any    all")
    for k in sorted({1, 3, top_k}):
        print(f"  @{k:<7}{recall(results, k):.2f}   {recall_all(results, k):.2f}")
    print("  any = at least one expected chunk retrieved")
    print("  all = every expected chunk retrieved")
    print(f"\nMRR@{top_k:<7} {mrr(results, top_k):.3f}")

    # Split by how the question is phrased. The headline number averages two
    # very different populations: questions using the Act's own vocabulary
    # match on terminology almost for free, because every chunk is embedded
    # with its citation header. Whether a retrieval change helped is almost
    # entirely a question about the lay group.
    groups = sorted({result.question.group for result in results})
    if len(groups) > 1:
        print(f"\ngroup        n   any@{top_k}  all@{top_k}  MRR@{top_k}")
        for group in groups:
            subset = [r for r in results if r.question.group == group]
            print(
                f"{group:<12} {len(subset):<3} {recall(subset, top_k):.2f}"
                f"   {recall_all(subset, top_k):.2f}   {mrr(subset, top_k):.3f}"
            )

    misses = [result for result in results if not result.recall_at(top_k)]
    if not misses:
        print("\nno misses")
    else:
        # Misses are printed with what came back instead, because an aggregate
        # number tells you retrieval got worse and nothing about where to look.
        print(f"\n{len(misses)} miss{'es' if len(misses) != 1 else ''}:")
        for result in misses:
            print(f"\n  {result.question.id}")
            print(f"    asked     {result.question.question}")
            print(f"    wanted    {', '.join(result.question.expect)}")
            got = ", ".join(provenance(hit) for hit in result.hits[:5])
            print(f"    got       {got or '(nothing)'}")

        # A gold chunk sitting just outside the cut is a different problem from
        # one retrieval cannot find at all: the first is a k or a re-ranking
        # question, the second is an embedding question.
        deep = [r for r in misses if r.gold_rank is not None]
        if deep:
            print(
                f"\n  {len(deep)} of those did retrieve a gold chunk, below the cut: "
                + ", ".join(f"{r.question.id}@{r.gold_rank}" for r in deep)
            )

    # Questions the loose metric calls a hit and the strict one a miss. These
    # are invisible in every number above and are the reason the strict metric
    # exists: whatever is listed here is a provision the answer cannot cite
    # because it was never retrieved.
    partial = [
        result
        for result in results
        if result.recall_at(top_k) and not result.covered_at(top_k)
    ]
    if not partial:
        return
    print(f"\n{len(partial)} partial (hit under any, miss under all):")
    for result in partial:
        found = len(result.question.expect) - len(result.missing_at(top_k))
        print(
            f"  {result.question.id:<34} {found}/{len(result.question.expect)}"
            f"  not retrieved: {', '.join(result.missing_at(top_k))}"
        )


def sweep(results_by_k: dict[int, list[Result]]) -> None:
    print("\nk    any    all    MRR")
    for k, results in sorted(results_by_k.items()):
        print(
            f"{k:<5}{recall(results, k):.2f}   {recall_all(results, k):.2f}"
            f"   {mrr(results, k):.3f}"
        )


def run_followups(
    items: list[Followup],
    generator: Generator,
    collection,
    bm25: Bm25Index,
    embedder,
    top_k: int,
    candidates: int,
) -> tuple[dict[str, list[Result]], dict[str, str]]:
    """Retrieve each follow-up as typed, as rewritten, and as its standalone.

    The rewrite goes through ``core.generate.rewrite``, the function the graph
    calls, so what is measured is what ships and not a copy of it.
    """
    results: dict[str, list[Result]] = {condition: [] for condition in CONDITIONS}
    rewrites: dict[str, str] = {}
    for item in items:
        rewrites[item.id] = rewrite(generator, item.question, item.history)
        texts = {
            "typed": item.question,
            "rewrite": rewrites[item.id],
            "standalone": item.standalone,
        }
        for condition, text in texts.items():
            # Wrapped as a Question so the single-turn metrics apply unchanged.
            question = Question(
                id=item.id,
                question=text,
                expect=item.expect,
                article_no="",
                group=condition,
            )
            hits = hybrid_search(collection, bm25, embedder, text, top_k, candidates)
            results[condition].append(Result(question=question, hits=hits))
    return results, rewrites


def report_followups(
    results: dict[str, list[Result]], rewrites: dict[str, str], top_k: int
) -> None:
    print(f"follow-ups {len(rewrites)}")
    print(f"retrieved  top {top_k}")
    print(f"\ncondition    any@{top_k}  all@{top_k}  MRR@{top_k}")
    for condition in CONDITIONS:
        subset = results[condition]
        print(
            f"{condition:<12} {recall(subset, top_k):.2f}"
            f"   {recall_all(subset, top_k):.2f}   {mrr(subset, top_k):.3f}"
        )
    print("  typed      = the follow-up as asked, the control")
    print("  standalone = the hand-written equivalent, the ceiling")

    # Per item, because the aggregate cannot say whether the rewrite lost to
    # the ceiling on a pronoun or on an echo of the previous turn's topic.
    print()
    for index, item_id in enumerate(rewrites):
        hit = [
            condition
            for condition in CONDITIONS
            if results[condition][index].recall_at(top_k)
        ]
        print(f"  {item_id:<32} hit: {', '.join(hit) or '(none)'}")
        print(f"    rewrite   {rewrites[item_id]}")
        lost = results["standalone"][index].recall_at(top_k) and not results["rewrite"][
            index
        ].recall_at(top_k)
        if lost:
            got = ", ".join(
                provenance(h) for h in results["rewrite"][index].hits[:top_k]
            )
            print(f"    LOST      rewrite missed what standalone found; got {got}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score retrieval over questions.yaml.")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="how many chunks to retrieve (default: TOP_K from the environment)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=f"also report recall and MRR at k = {', '.join(map(str, SWEEP))}",
    )
    parser.add_argument(
        "--followups",
        action="store_true",
        help="score the follow-up rewrite over followups.yaml instead",
    )
    parser.add_argument(
        "--floor",
        action="store_true",
        help="report best dense similarity, in scope vs offtopic.yaml",
    )
    args = parser.parse_args()

    config = load_config()
    top_k = args.top_k if args.top_k is not None else config.top_k

    if args.followups:
        return main_followups(config, top_k)
    if args.floor:
        return main_floor(config, top_k)

    questions = load_questions()
    collection = open_collection(config.index_dir, config.scratch_dir)
    bm25 = open_bm25(config.chunks_dir)
    check_gold_exists(questions, collection)
    embedder = openai_embedder(config.openai_api_key, config.embed_model)

    results = run(
        questions, collection, bm25, embedder, top_k, config.fusion_candidates
    )
    report(results, top_k)

    if args.sweep:
        # Retrieved once at the largest k and truncated, rather than querying
        # five times: the ranking is identical and it is four fewer API calls
        # per question.
        widest = run(
            questions,
            collection,
            bm25,
            embedder,
            max(SWEEP),
            config.fusion_candidates,
        )
        sweep({k: widest for k in SWEEP})

    return 0


def main_followups(config, top_k: int) -> int:
    items = load_followups()
    collection = open_collection(config.index_dir, config.scratch_dir)
    bm25 = open_bm25(config.chunks_dir)
    check_gold_exists(
        [
            Question(
                id=i.id, question=i.question, expect=i.expect, article_no="", group=""
            )
            for i in items
        ],
        collection,
    )
    embedder = openai_embedder(config.openai_api_key, config.embed_model)
    # One rewrite per item and no more, across the whole invocation.
    generator = CappedGenerator(
        inner=openai_generator(
            config.openai_api_key, config.chat_model, config.temperature
        ),
        limit=len(items),
    )

    results, rewrites = run_followups(
        items,
        generator,
        collection,
        bm25,
        embedder,
        top_k,
        config.fusion_candidates,
    )
    report_followups(results, rewrites, top_k)
    return 0


def best_dense(hits: list[Hit]) -> float | None:
    """The number core.generate.answer compares against MIN_DENSE_SCORE."""
    return max((h.dense_score for h in hits if h.dense_score is not None), default=None)


def main_floor(config, top_k: int) -> int:
    """Where MIN_DENSE_SCORE should sit, measured rather than chosen.

    Embedding calls only; nothing here generates. In scope is every question in
    questions.yaml plus the standalone form of every follow-up, since the graph
    answers the rewrite rather than what was typed.
    """
    in_scope = [q.question for q in load_questions()]
    in_scope += [item.standalone for item in load_followups()]
    offtopic = yaml.safe_load(OFFTOPIC_PATH.read_text(encoding="utf-8"))

    collection = open_collection(config.index_dir, config.scratch_dir)
    bm25 = open_bm25(config.chunks_dir)
    embedder = openai_embedder(config.openai_api_key, config.embed_model)

    def score(question: str) -> float:
        hits = hybrid_search(
            collection, bm25, embedder, question, top_k, config.fusion_candidates
        )
        best = best_dense(hits)
        return -1.0 if best is None else best

    scope_scores = sorted(score(q) for q in in_scope)
    off_scores = [(entry, score(entry["question"])) for entry in offtopic]

    print(f"in scope, n={len(scope_scores)}, lowest five:")
    print("    " + "  ".join(f"{s:.3f}" for s in scope_scores[:5]))
    print(f"\noff topic, n={len(off_scores)}, highest first:")
    for entry, s in sorted(off_scores, key=lambda pair: -pair[1]):
        print(f"    {s:.3f}  {entry['kind']:<9} {entry['id']}")

    print("\n    floor  in-scope refused  unrelated+probe passed  adjacent passed")
    for floor in FLOORS:
        refused = sum(s < floor for s in scope_scores)
        passed = sum(
            s >= floor for entry, s in off_scores if entry["kind"] != "adjacent"
        )
        adjacent = sum(
            s >= floor for entry, s in off_scores if entry["kind"] == "adjacent"
        )
        print(f"    {floor:.3f}  {refused:>16}  {passed:>22}  {adjacent:>15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
