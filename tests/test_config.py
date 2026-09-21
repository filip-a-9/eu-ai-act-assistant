"""Policy contract for configuration.

These are regression guards, not test-driven design: ``core/config.py`` already
satisfies every assertion here. They exist because each one pins a Phase 0
decision that is cheap to undo by accident -- a ``.get()`` with a default
quietly replacing ``_require``, or a ``field(repr=False)`` lost in a refactor
that then prints the API key into the per-query log CLAUDE.md mandates.

Nothing here reads the developer's real ``.env``. ``load_config()`` calls
``load_dotenv`` unconditionally, so a test for "the key is missing" would
otherwise pass on a machine with no ``.env`` and fail on the machine that
actually has one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import config as config_module
from core.config import PROJECT_ROOT, Config, load_config

# Every variable load_config() consults. Cleared wholesale so a test never
# inherits a value from the shell that ran pytest.
CONFIG_VARS = (
    "OPENAI_API_KEY",
    "EMBED_MODEL",
    "CHAT_MODEL",
    "TOP_K",
    "TEMPERATURE",
    "RAW_DIR",
    "CHUNKS_DIR",
    "INDEX_DIR",
    "LOGS_DIR",
)


@pytest.fixture
def isolated_env(monkeypatch):
    """A process environment with no config in it and no .env on disk.

    ``load_dotenv`` is stubbed rather than pointed at a temporary file: the
    behaviour under test is how ``Config`` reads the environment, and the
    dotenv library's own file discovery is not ours to re-test.
    """
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: False)
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Required values fail loudly
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_an_error_naming_the_variable(isolated_env):
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    assert "OPENAI_API_KEY" in str(excinfo.value)


def test_empty_api_key_is_treated_as_missing(isolated_env):
    # An exported-but-blank variable is the classic CI failure: `os.environ`
    # contains the key, so a truthiness check is the only thing that catches it.
    isolated_env.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    assert "OPENAI_API_KEY" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Malformed values fail loudly rather than defaulting
# ---------------------------------------------------------------------------


def test_non_numeric_top_k_fails_instead_of_silently_defaulting(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("TOP_K", "banana")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "TOP_K" in message and "banana" in message


def test_non_numeric_temperature_fails_instead_of_silently_defaulting(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("TEMPERATURE", "warm")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    assert "TEMPERATURE" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Secret hygiene -- every query gets logged, so repr() is a leak surface
# ---------------------------------------------------------------------------


def test_repr_does_not_contain_the_api_key(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-do-not-print-me")
    assert "sk-do-not-print-me" not in repr(load_config())


def test_the_api_key_is_still_reachable_on_the_object(isolated_env):
    # The companion to the test above: hiding the key from repr() is only
    # correct if the key is still there to use.
    isolated_env.setenv("OPENAI_API_KEY", "sk-do-not-print-me")
    assert load_config().openai_api_key == "sk-do-not-print-me"


# ---------------------------------------------------------------------------
# Defaults and paths
# ---------------------------------------------------------------------------


def test_defaults_match_the_documented_stack(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    assert cfg.embed_model == "text-embedding-3-small"
    assert cfg.chat_model == "gpt-5.6-terra"
    assert cfg.top_k == 5
    # Not 0.0: gpt-5.6 rejects every temperature but its default with a 400.
    assert cfg.temperature == 1.0


def test_relative_paths_anchor_to_the_repo_root_not_the_working_directory(
    isolated_env,
):
    # So `python scripts/build_index.py` writes the same place from anywhere.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("INDEX_DIR", "data/index")
    assert load_config().index_dir == PROJECT_ROOT / "data" / "index"


def test_absolute_paths_are_left_alone(isolated_env, tmp_path):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("INDEX_DIR", str(tmp_path))
    assert load_config().index_dir == tmp_path


def test_config_is_frozen(isolated_env):
    # Config is read by retrieval and generation on every query; a module that
    # mutates top_k in passing would be very hard to find.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    with pytest.raises(Exception):
        cfg.top_k = 99  # type: ignore[misc]


def test_every_path_on_the_config_is_a_path_object(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    assert isinstance(cfg.raw_dir, Path)
    assert isinstance(cfg.chunks_dir, Path)
    assert isinstance(cfg.index_dir, Path)


def test_config_carries_no_fields_beyond_the_documented_set():
    # A guard against configuration sprawl: anything added here should be a
    # deliberate decision recorded in DECISIONS.md, not an incidental field.
    assert set(Config.__dataclass_fields__) == {
        "openai_api_key",
        "embed_model",
        "chat_model",
        "top_k",
        "temperature",
        "raw_dir",
        "chunks_dir",
        "index_dir",
        "logs_dir",
    }
