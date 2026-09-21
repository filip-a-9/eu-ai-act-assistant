"""Structural contract for the persisted index.

Every test builds a throwaway Chroma directory under ``tmp_path`` from the
six-chunk corpus in ``conftest.py`` and a fake embedder. Nothing here measures
retrieval quality -- that is ``evals/run_eval.py``'s job -- and nothing calls
OpenAI.

Two of these guard traps that fail silently rather than loudly: a collection
built in the wrong distance space still returns five results, and a collection
carrying Chroma's default embedding function still answers queries. Both give
worse answers with no error anywhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.index import COLLECTION_NAME, build, load_chunk_records, open_collection


class RecordingEmbedder:
    """A fake that remembers what it was asked to embed."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]


# ---------------------------------------------------------------------------
# What lands in the collection
# ---------------------------------------------------------------------------


def test_every_record_is_stored(built_index, tiny_corpus):
    assert open_collection(built_index).count() == len(tiny_corpus)


def test_chunk_ids_are_the_collection_keys(built_index, tiny_corpus):
    # The eval scores against chunk ids, so this is the contract it rests on.
    stored = open_collection(built_index).get(include=[])["ids"]
    assert sorted(stored) == sorted(record["id"] for record in tiny_corpus)


def test_the_stored_document_is_the_clean_law_not_the_embedded_text(
    built_index, tiny_corpus
):
    # embed_text carries an editorial citation header that makes a short chunk
    # findable. The UI has to quote the law without that header welded on, so
    # the document stored for display must be `text`.
    #
    # strict: the assertion is inside the loop, so a short `documents` list
    # would run the body zero times and report green having checked nothing.
    expected = {record["id"]: record["text"] for record in tiny_corpus}
    stored = open_collection(built_index).get(include=["documents"])
    for chunk_id, document in zip(stored["ids"], stored["documents"], strict=True):
        assert document == expected[chunk_id]


def test_the_embedded_text_is_the_citation_header_plus_the_law(tmp_path, tiny_corpus):
    embedder = RecordingEmbedder()
    build(tiny_corpus, embedder, tmp_path / "index")
    assert embedder.seen == [record["embed_text"] for record in tiny_corpus]


def test_metadata_round_trips_with_every_value_a_string(built_index):
    # Chroma drops keys whose value is None, which would turn a missing
    # citation into a KeyError at answer time instead of a failure here.
    stored = open_collection(built_index).get(include=["metadatas"])
    for metadata in stored["metadatas"]:
        assert metadata["article_no"]
        for key, value in metadata.items():
            assert isinstance(value, str), f"{key} is {type(value).__name__}"


def test_metadata_carries_every_field_the_chunker_produces(built_index):
    stored = open_collection(built_index).get(include=["metadatas"])
    expected = {
        "kind",
        "article_no",
        "number",
        "paragraph",
        "parent_id",
        "title",
        "chapter",
        "source_url",
    }
    for metadata in stored["metadatas"]:
        assert set(metadata) == expected


# ---------------------------------------------------------------------------
# The two silent traps
# ---------------------------------------------------------------------------

# Passing embedding_function=None makes Chroma 1.5.9 persist the marker
# {"type": "legacy"}, and it warns every time that config is read back. The
# build, open and query path never reads it and is warning-free (verified);
# only these three tests look, so the filter is applied here rather than
# globally, where it would hide the warning appearing somewhere new.
_LEGACY_EF_WARNING = pytest.mark.filterwarnings(
    "ignore:legacy embedding function config:DeprecationWarning"
)


@_LEGACY_EF_WARNING
def test_the_collection_uses_cosine_distance(built_index):
    # Chroma falls back to l2 when no space is declared. OpenAI embeddings are
    # normalised for cosine; built in l2 the index still returns five results,
    # just worse ones, and nothing raises.
    configuration = open_collection(built_index).configuration
    assert configuration["hnsw"]["space"] == "cosine"


@_LEGACY_EF_WARNING
def test_no_default_embedding_function_is_attached(built_index):
    # Chroma's default is ONNX MiniLM. Left in place it will happily answer a
    # query_texts= call by embedding the question in a different vector space
    # from the documents -- garbage rankings, no error. It also pulls an ~80 MB
    # model down on first use, which is a cold-start failure on a deployed host
    # and never reproduces locally.
    assert open_collection(built_index).configuration["embedding_function"] is None


@_LEGACY_EF_WARNING
def test_querying_by_text_fails_rather_than_silently_mixing_vector_spaces(
    built_index,
):
    # The consequence of the test above, stated as behaviour: there is no way
    # to accidentally get an answer out of the wrong embedding space.
    #
    # Chroma 1.5.9 raises ValueError here, checked by running it rather than
    # guessing: "You must provide an embedding function to compute embeddings".
    # The type alone is too broad to carry the claim -- a bad n_results would
    # also be a ValueError -- so the message fragment is what distinguishes
    # "refused to embed the question" from "rejected the call for some other
    # reason". Only the stable clause is matched, not the whole string, which
    # also carries a documentation URL that is free to change.
    with pytest.raises(ValueError, match="embedding function"):
        open_collection(built_index).query(query_texts=["prohibited"], n_results=1)


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------


def test_rebuilding_replaces_rather_than_duplicating(
    tmp_path, tiny_corpus, fake_embedder
):
    index_dir = tmp_path / "index"
    build(tiny_corpus, fake_embedder, index_dir)
    build(tiny_corpus, fake_embedder, index_dir)
    assert open_collection(index_dir).count() == len(tiny_corpus)


def test_rebuilding_drops_records_that_are_no_longer_in_the_corpus(
    tmp_path, tiny_corpus, fake_embedder
):
    index_dir = tmp_path / "index"
    build(tiny_corpus, fake_embedder, index_dir)
    build(tiny_corpus[:2], fake_embedder, index_dir)
    collection = open_collection(index_dir)
    assert collection.count() == 2
    assert "rct_27" not in collection.get(include=[])["ids"]


def test_building_an_empty_corpus_yields_an_empty_collection(tmp_path, fake_embedder):
    build([], fake_embedder, tmp_path / "index")
    assert open_collection(tmp_path / "index").count() == 0


def test_rebuilding_leaves_no_orphaned_index_segments(
    tmp_path, tiny_corpus, fake_embedder
):
    # Chroma's delete_collection drops the collection from sqlite but leaves
    # its HNSW segment directory on disk. The index is committed, so an
    # orphan per rebuild is ~600 KB of dead binary added to git history
    # forever, and the directory grows without bound.
    index_dir = tmp_path / "index"
    for _ in range(3):
        build(tiny_corpus, fake_embedder, index_dir)
    segments = [path for path in index_dir.iterdir() if path.is_dir()]
    assert len(segments) == 1


def test_building_refuses_to_clobber_a_directory_that_is_not_an_index(
    tmp_path, tiny_corpus, fake_embedder
):
    # A full rebuild deletes the target directory, so a mistyped INDEX_DIR
    # must fail rather than take someone's files with it.
    target = tmp_path / "not-an-index"
    target.mkdir()
    (target / "important.txt").write_text("do not delete me", encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        build(tiny_corpus, fake_embedder, target)
    assert "important.txt" not in str(excinfo.value)  # no contents leaked
    assert (target / "important.txt").exists()


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_build_reports_progress_as_it_goes(tmp_path, tiny_corpus, fake_embedder):
    # A real build makes nine paid API calls over about a minute. Without this
    # the script is silent throughout, and a hang is indistinguishable from
    # normal slowness.
    seen = []
    build(
        tiny_corpus,
        fake_embedder,
        tmp_path / "index",
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (len(tiny_corpus), len(tiny_corpus))


def test_build_without_a_progress_callback_still_works(
    tmp_path, tiny_corpus, fake_embedder
):
    assert build(tiny_corpus, fake_embedder, tmp_path / "index") == len(tiny_corpus)


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def test_opening_a_missing_index_names_the_script_that_builds_it(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        open_collection(tmp_path / "nothing-here")
    assert "build_index.py" in str(excinfo.value)


def test_opening_a_directory_with_no_collection_names_the_script_too(tmp_path):
    # A half-built index -- the directory exists, the collection does not --
    # is the shape a cancelled build leaves behind.
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with pytest.raises(RuntimeError) as excinfo:
        open_collection(index_dir)
    assert "build_index.py" in str(excinfo.value)


def test_the_collection_name_is_stable(built_index):
    # Renaming it silently orphans every previously built index.
    assert open_collection(built_index).name == COLLECTION_NAME


# ---------------------------------------------------------------------------
# Opening must not write to the index it was given
# ---------------------------------------------------------------------------

# The index is committed to the repository, and Chroma's persistent client is a
# read-write database rather than a file reader: connecting opens a write
# transaction, and the first read flushes an HNSW buffer. Neither settles --
# part of length.bin is serialised heap pointers, which ASLR re-randomises
# every run -- so there is no stable version that could be committed once to
# make it stop. These three pin the behaviour that keeps the committed index
# untouched and lets a read-only host open it at all.


def _tree_digest(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, keyed by relative path."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def chmod_tree():
    """Set a mode on every file under a directory, restoring it afterwards.

    Restoring matters: pytest's own ``tmp_path`` cleanup cannot remove a tree
    it has no permission to write, and that failure surfaces attached to
    whichever test runs next, which is a confusing place to start looking.
    """
    touched: list[tuple[Path, int]] = []

    def _apply(root: Path, mode: int) -> Path:
        for path in Path(root).rglob("*"):
            if path.is_file():
                touched.append((path, path.stat().st_mode))
                path.chmod(mode)
        return root

    yield _apply
    for path, original in reversed(touched):
        path.chmod(original)


def test_opening_and_reading_does_not_modify_the_index_directory(built_index):
    before = _tree_digest(built_index)
    open_collection(built_index).count()
    assert _tree_digest(built_index) == before


def test_an_index_whose_files_are_read_only_still_opens(
    built_index, tiny_corpus, chmod_tree
):
    # A container with a read-only rootfs is a common deployment default. The
    # app is required to open a prebuilt index read-only, so this is that rule
    # stated as something a machine can check.
    collection = open_collection(chmod_tree(built_index, 0o444))
    assert collection.count() == len(tiny_corpus)


def test_a_permission_failure_is_not_reported_as_a_cancelled_build(
    built_index, chmod_tree
):
    # An unreadable index cannot be rescued by copying it anywhere, so this one
    # still has to fail -- the assertion is about *what it says*. Blaming a
    # cancelled build sends someone to rebuild an index that is present and
    # intact, which on a locked-down host is an expensive wrong turn.
    with pytest.raises(RuntimeError) as excinfo:
        open_collection(chmod_tree(built_index, 0o000))
    assert "build_index.py" not in str(excinfo.value)


def test_a_failed_open_leaves_no_scratch_copy_behind(built_index, chmod_tree, tmp_path):
    # The copy is made before it can be known to succeed, so the failure path
    # owns the half-built directory. Deferring it to interpreter exit is not
    # enough: a long-lived Streamlit process that retries a failing open would
    # accumulate one abandoned copy per attempt.
    scratch = tmp_path / "scratch"
    with pytest.raises(RuntimeError):
        open_collection(chmod_tree(built_index, 0o000), scratch)
    assert list(scratch.iterdir()) == []


# ---------------------------------------------------------------------------
# Reading the corpus off disk
# ---------------------------------------------------------------------------


def test_load_chunk_records_reads_one_record_per_line(tmp_path, tiny_corpus):
    import json

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    with (chunks_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in tiny_corpus:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    assert load_chunk_records(chunks_dir) == tiny_corpus


def test_load_chunk_records_names_the_script_that_writes_the_file(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        load_chunk_records(tmp_path / "nothing-here")
    assert "build_chunks.py" in str(excinfo.value)
