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


def _as_optional_path(name: str) -> Path | None:
    """Like ``_as_path``, but unset means None rather than a baked-in default.

    For settings whose sensible default is decided by the platform rather than
    by this repo, where a literal here would be a guess at a host's layout.
    """
    raw = os.environ.get(name)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Config:
    # repr=False keeps the key out of tracebacks and out of the per-query logs
    # that CLAUDE.md requires.
    openai_api_key: str = field(repr=False)
    embed_model: str
    chat_model: str
    top_k: int
    temperature: float
    max_questions: int
    max_questions_per_ip: int
    raw_dir: Path
    chunks_dir: Path
    index_dir: Path
    # Where the index is copied before it is opened. None means the system
    # temp directory. See core/index.py::_writable_copy for why a copy exists.
    scratch_dir: Path | None
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
        # The Streamlit app's per-session spend bound. A number here rather
        # than a literal in app.py, for the same reason as top_k: the value a
        # deployment needs to change is the one thing it must not have to edit
        # code to change.
        max_questions=_as_int("MAX_QUESTIONS", "10"),
        # The same bound across sessions from one address, so that reloading
        # the page stops being a way around the line above. Three sessions'
        # worth rather than one: an office or a mobile carrier reaches the app
        # from a single address, and capping that at 10 would lock out everyone
        # behind the first visitor. Not a security control -- the address comes
        # from the connection and can be spoofed.
        max_questions_per_ip=_as_int("MAX_QUESTIONS_PER_IP", "30"),
        raw_dir=_as_path("RAW_DIR", "data/raw"),
        chunks_dir=_as_path("CHUNKS_DIR", "data/chunks"),
        index_dir=_as_path("INDEX_DIR", "data/index"),
        # Unset on purpose: the right writable location is a property of the
        # host, and a default here would be this repo guessing at one.
        scratch_dir=_as_optional_path("SCRATCH_DIR"),
        logs_dir=_as_path("LOGS_DIR", "data/logs"),
    )
