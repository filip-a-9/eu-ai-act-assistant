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

from dataclasses import FrozenInstanceError
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
    "FUSION_CANDIDATES",
    "TEMPERATURE",
    "MAX_QUESTIONS",
    "MAX_QUESTIONS_PER_IP",
    "MIN_DENSE_SCORE",
    "LOG_SALT",
    "LOG_RETENTION_DAYS",
    "RAW_DIR",
    "CHUNKS_DIR",
    "INDEX_DIR",
    "SCRATCH_DIR",
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


def test_non_numeric_fusion_candidates_fails_instead_of_silently_defaulting(
    isolated_env,
):
    # How deep each retriever goes before the rankings are fused. A typo that
    # fell back to the default would change what retrieval returns while every
    # test and eval still passed, which is the failure this project can least
    # afford to have be silent.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("FUSION_CANDIDATES", "deep")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "FUSION_CANDIDATES" in message and "deep" in message


def test_non_numeric_max_questions_fails_instead_of_silently_defaulting(isolated_env):
    # This one bounds spending, so a typo that fell back to the default would
    # be a silently larger bill than the deployment intended.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("MAX_QUESTIONS", "lots")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "MAX_QUESTIONS" in message and "lots" in message


def test_non_numeric_max_questions_per_ip_fails_instead_of_silently_defaulting(
    isolated_env,
):
    # The wider of the two caps, and the one a deployment is most likely to
    # retune, so the same argument applies: falling back to the default on a
    # typo would spend more than the deployment asked for.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("MAX_QUESTIONS_PER_IP", "plenty")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "MAX_QUESTIONS_PER_IP" in message and "plenty" in message


def test_non_numeric_min_dense_score_fails_instead_of_silently_defaulting(
    isolated_env,
):
    # A typo falling back to the default would move the line between answering
    # and refusing without anyone having decided to.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("MIN_DENSE_SCORE", "high")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "MIN_DENSE_SCORE" in message and "high" in message


def test_non_numeric_log_retention_fails_instead_of_silently_defaulting(
    isolated_env,
):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("LOG_RETENTION_DAYS", "forever")
    with pytest.raises(RuntimeError) as excinfo:
        load_config()
    message = str(excinfo.value)
    assert "LOG_RETENTION_DAYS" in message and "forever" in message


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


def test_repr_does_not_contain_the_log_salt(isolated_env):
    # The salt is what keeps a hashed address from being reversed by hashing
    # every IPv4 address; printed into a traceback it protects nothing.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("LOG_SALT", "salt-do-not-print-me")
    assert "salt-do-not-print-me" not in repr(load_config())


def test_an_unset_log_salt_is_none(isolated_env):
    # None, never a baked-in salt: a salt committed to the repository is one
    # anybody can hash every address with.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    assert load_config().log_salt is None


# ---------------------------------------------------------------------------
# Defaults and paths
# ---------------------------------------------------------------------------


def test_defaults_match_the_documented_stack(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    assert cfg.embed_model == "text-embedding-3-small"
    assert cfg.chat_model == "gpt-5.6-terra"
    assert cfg.top_k == 5
    # Ten times top_k: deep enough that a chunk one retriever ranks poorly can
    # still be rescued by the other, shallow enough that fusion stays cheap.
    assert cfg.fusion_candidates == 50
    # Not 0.0: gpt-5.6 rejects every temperature but its default with a 400.
    assert cfg.temperature == 1.0
    # The per-session question cap the Streamlit app enforces before it spends.
    assert cfg.max_questions == 10
    # Three sessions' worth: wide enough that an office or a mobile carrier
    # behind one address is not locked out by its first visitor.
    assert cfg.max_questions_per_ip == 30
    # Measured by `run_eval.py --floor`: the lowest in-scope question scores
    # 0.190 and the highest unrelated one 0.177. See DECISIONS.md.
    assert cfg.min_dense_score == 0.18
    assert cfg.log_retention_days == 30


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


def test_the_scratch_directory_is_unset_by_default(isolated_env):
    # None, not a path: core/index.py copies the index somewhere writable
    # before opening it, and unset means "wherever tempfile puts things on this
    # platform". Baking a default in here would guess at a host's layout.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    assert load_config().scratch_dir is None


def test_the_scratch_directory_can_be_pointed_at_a_writable_path(
    isolated_env, tmp_path
):
    # The setting a locked-down host actually needs: the one writable mount it
    # has may not be the system temp directory.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("SCRATCH_DIR", str(tmp_path))
    assert load_config().scratch_dir == tmp_path


def test_config_is_frozen(isolated_env):
    # Config is read by retrieval and generation on every query; a module that
    # mutates top_k in passing would be very hard to find.
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    # FrozenInstanceError specifically, not Exception: a bare Exception here
    # also passes on an AttributeError from a renamed field and a TypeError
    # from a changed signature, so it would claim to pin immutability while
    # really reporting that *something* went wrong.
    with pytest.raises(FrozenInstanceError):
        cfg.top_k = 99  # type: ignore[misc]


def test_every_path_on_the_config_is_a_path_object(isolated_env):
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    assert isinstance(cfg.raw_dir, Path)
    assert isinstance(cfg.chunks_dir, Path)
    assert isinstance(cfg.index_dir, Path)


def test_every_variable_load_config_reads_is_cleared_by_the_fixture():
    """CONFIG_VARS must list every env var ``core/config.py`` consults.

    Derived from the module's source rather than restated by hand, because a
    hand-maintained list is what failed: ``MAX_QUESTIONS_PER_IP`` and
    ``SCRATCH_DIR`` were absent, so two tests above read whatever the shell
    running pytest happened to export and passed or failed for reasons that
    had nothing to do with the code.
    """
    import ast

    source = (PROJECT_ROOT / "core" / "config.py").read_text(encoding="utf-8")
    readers = {"_require", "_as_int", "_as_float", "_as_path", "_as_optional_path"}

    consulted = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        # The helpers above, plus os.environ.get() where config reads a var
        # directly. A non-literal first argument is the helper's own body
        # forwarding a name it was given, which is not a variable itself.
        if name in readers or name == "get":
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                consulted.add(first.value)

    assert consulted, "found no environment variables; the parser is wrong"
    assert consulted <= set(CONFIG_VARS), (
        "these variables are read by core/config.py but not cleared by the "
        f"isolated_env fixture: {sorted(consulted - set(CONFIG_VARS))}"
    )


def test_config_carries_no_fields_beyond_the_documented_set():
    # A guard against configuration sprawl: anything added here should be a
    # deliberate decision recorded in DECISIONS.md, not an incidental field.
    assert set(Config.__dataclass_fields__) == {
        "openai_api_key",
        "embed_model",
        "chat_model",
        "top_k",
        "fusion_candidates",
        "temperature",
        "max_questions",
        "max_questions_per_ip",
        "min_dense_score",
        "log_salt",
        "log_retention_days",
        "raw_dir",
        "chunks_dir",
        "index_dir",
        "scratch_dir",
        "logs_dir",
    }
