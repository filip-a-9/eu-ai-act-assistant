"""Shared fixtures. Assertions never live here.

Every test reads HTML from ``tests/fixtures/`` rather than ``data/raw/``, which
is gitignored: a suite that depended on the fetched corpus would not run on a
fresh clone. No test in this file or any other touches the network.
"""

from __future__ import annotations

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
