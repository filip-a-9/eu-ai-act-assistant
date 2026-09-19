"""Shared fixtures. Assertions never live here.

Every test reads HTML from ``tests/fixtures/`` rather than ``data/raw/``, which
is gitignored: a suite that depended on the fetched corpus would not run on a
fresh clone. No test in this file or any other touches the network.

The retrieval fixtures below follow the same rule from the other direction:
a fake embedder replaces OpenAI, and the corpus is six hand-written chunks
whose right answers can be stated by eye. Loading the real 886-chunk index
here would test quality, which is ``evals/run_eval.py``'s job, not pytest's.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_html():
    """Return a loader for a named fixture under ``tests/fixtures/``."""

    def _load(name: str) -> str:
        path = FIXTURE_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"missing fixture: {path}")
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture(scope="session")
def chunks_by_id():
    """Index a list of chunks by id, asserting nothing."""

    def _index(chunks):
        return {c.id: c for c in chunks}

    return _index


# ---------------------------------------------------------------------------
# Retrieval fixtures
# ---------------------------------------------------------------------------

# A fixed vocabulary, so a vector means the same thing in every test. These are
# the distinctive terms of the tiny corpus below; a query word outside the list
# simply contributes nothing, which is the behaviour a real embedder would
# approximate for a term it has no signal for.
VOCABULARY = (
    "prohibited", "practices", "manipulate", "social", "scoring", "biometric",
    "high", "risk", "classification", "annex", "safety", "component",
    "management", "system", "iterative", "lifecycle", "documented",
    "penalties", "fines", "administrative", "million", "turnover",
    "trustworthy", "human", "oversight", "transparency", "principles",
    "education", "employment", "law", "enforcement", "migration",
)

_WORD = re.compile(r"[a-z]+")


class FakeEmbedder:
    """Deterministic bag-of-words vectors over ``VOCABULARY``, L2-normalised.

    Deliberately not a hash of the text. Hashed vectors are deterministic but
    arbitrary, so an assertion about *which* chunk ranks first would be a
    restatement of the hash rather than a statement about retrieval. With a
    bag of words, "what practices are prohibited" really is nearest the chunk
    about prohibited practices, and a broken query path really does show up.

    The trailing constant dimension keeps a vector with no vocabulary hits
    from being all zeros, which cosine distance reports as NaN.
    """

    dimensions = len(VOCABULARY) + 1

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        words = set(_WORD.findall(text.lower()))
        raw = [1.0 if term in words else 0.0 for term in VOCABULARY] + [0.1]
        length = math.sqrt(sum(value * value for value in raw))
        return [value / length for value in raw]


def _record(chunk_id, kind, article_no, number, paragraph, parent_id,
            title, chapter, text):
    """Build one chunk record in the shape build_chunks.py writes."""
    return {
        "id": chunk_id,
        "kind": kind,
        "article_no": article_no,
        "number": number,
        "paragraph": paragraph,
        "parent_id": parent_id,
        "title": title,
        "chapter": chapter,
        "source_url": f"https://example.invalid/eli#{parent_id or chunk_id}",
        "text": text,
        # Mirrors the real chunker: the citation header is part of what gets
        # embedded, which is what makes a short chunk findable at all.
        "embed_text": f"{article_no} — {title}\n\n{text}",
    }


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def tiny_corpus():
    """Six records whose correct answers are obvious by inspection."""
    return [
        _record(
            "art_5.para_1", "article", "Article 5(1)", "5", "1", "art_5",
            "Prohibited AI practices", "Chapter II — PROHIBITED AI PRACTICES",
            "The following AI practices shall be prohibited: practices that "
            "manipulate a person, and social scoring of natural persons.",
        ),
        _record(
            "art_6.para_1", "article", "Article 6(1)", "6", "1", "art_6",
            "Classification rules for high-risk AI systems",
            "Chapter III — HIGH-RISK AI SYSTEMS",
            "An AI system is high risk where it is intended to be used as a "
            "safety component of a product covered by Annex I.",
        ),
        _record(
            "art_9.para_1", "article", "Article 9(1)", "9", "1", "art_9",
            "Risk management system", "Chapter III — HIGH-RISK AI SYSTEMS",
            "A risk management system shall be established for high risk AI "
            "systems as a continuous iterative process across the lifecycle "
            "and shall be documented.",
        ),
        _record(
            "art_99.para_3", "article", "Article 99(3)", "99", "3", "art_99",
            "Penalties", "Chapter XII — PENALTIES",
            "Non-compliance with the prohibited practices shall be subject to "
            "administrative fines of up to 35 million EUR or 7 % of turnover.",
        ),
        _record(
            "anx_III.sec_1", "annex", "Annex III, Section 1", "III",
            "Section 1", "anx_III", "High-risk AI systems referred to in "
            "Article 6(2)", "",
            "Areas of high risk use: biometric identification, education, "
            "employment, law enforcement and migration.",
        ),
        _record(
            "rct_27", "recital", "Recital 27", "27", "", "",
            "Recital 27", "",
            "Trustworthy AI rests on principles of human oversight and "
            "transparency, which guide the drawing up of codes of conduct.",
        ),
    ]


@pytest.fixture
def built_index(tmp_path, tiny_corpus, fake_embedder):
    """A persisted Chroma index over the tiny corpus, in a throwaway directory."""
    from core.index import build

    index_dir = tmp_path / "index"
    build(tiny_corpus, fake_embedder, index_dir)
    return index_dir
