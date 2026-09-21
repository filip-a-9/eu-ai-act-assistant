"""Single source of truth for configuration.

Nothing else in this repo reads ``os.environ``. Model names, ``top_k`` and
``temperature`` live here rather than as literals in retrieval or generation
code, so an eval sweep only has to change one thing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Paths resolve against the repo root, not the current working directory, so
# `python scripts/fetch_source.py` behaves the same no matter where it is run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _require(name: str) -> str:
    """Read a mandatory env var, or fail with a message naming which one."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def _as_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _as_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _as_path(name: str, default: str) -> Path:
    """Relative values anchor to the repo root; absolute values pass through."""
    raw = Path(os.environ.get(name, default))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


@dataclass(frozen=True)
class Config:
    # repr=False keeps the key out of tracebacks and out of the per-query logs
    # that CLAUDE.md requires.
    openai_api_key: str = field(repr=False)
    embed_model: str
    chat_model: str
    top_k: int
    temperature: float
    raw_dir: Path
    chunks_dir: Path
    index_dir: Path
    logs_dir: Path


def raw_dir() -> Path:
    """Where the fetched corpus lives.

    Deliberately separate from ``load_config()``: downloading public EUR-Lex
    HTML needs no OpenAI key, and requiring one to run the fetch script would
    be a pointless barrier on a fresh clone.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return _as_path("RAW_DIR", "data/raw")


def chunks_dir() -> Path:
    """Where the parsed corpus lands.

    Keyless for the same reason as ``raw_dir()``: parsing local HTML into
    chunks calls no model, so ``scripts/build_chunks.py`` must run on a fresh
    clone without an OpenAI key.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return _as_path("CHUNKS_DIR", "data/chunks")


def load_config() -> Config:
    """Build a Config from the environment, reading .env first if present."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Config(
        openai_api_key=_require("OPENAI_API_KEY"),
        embed_model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        chat_model=os.environ.get("CHAT_MODEL", "gpt-5.6-terra"),
        top_k=_as_int("TOP_K", "5"),
        # 1.0, not 0.0: gpt-5.6 accepts only its default temperature and
        # returns a 400 for anything else. Answers are held to the law by the
        # citation check in core/generate.py, not by a sampling parameter.
        temperature=_as_float("TEMPERATURE", "1.0"),
        raw_dir=_as_path("RAW_DIR", "data/raw"),
        chunks_dir=_as_path("CHUNKS_DIR", "data/chunks"),
        index_dir=_as_path("INDEX_DIR", "data/index"),
        logs_dir=_as_path("LOGS_DIR", "data/logs"),
    )