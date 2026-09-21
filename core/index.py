"""Build and open the persisted Chroma collection.

Two responsibilities, kept in one module because they share the collection's
name and its configuration, and disagreeing about either is the kind of bug
that produces an empty result set with no error:

* ``build`` runs offline from ``scripts/build_index.py`` and needs an API key;
* ``open_collection`` runs at query time and needs only the directory.

Two pieces of Chroma behaviour are load-bearing here, and both fail silently
rather than raising, so both are pinned by tests:

* **The distance space defaults to l2.** OpenAI embeddings are normalised for
  cosine. An l2 collection still returns results, just worse ones.
* **The embedding function defaults to ONNX MiniLM.** Every call passes
  ``embedding_function=None`` because this repo supplies its own vectors.
  Left in place, a ``query_texts=`` call would embed the question with MiniLM
  and compare it against OpenAI document vectors -- two unrelated spaces, no
  error -- and would download an ~80 MB model the first time it ran, which is
  a cold-start failure on a deployed host that never reproduces locally.
"""

from __future__ import annotations

import atexit
import json
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb

from core.embed import Embedder

COLLECTION_NAME = "ai_act"

# Written beside the index by scripts/build_index.py. Named here because
# _clear() has to recognise it as part of the index rather than as a stray
# file it must refuse to delete.
MANIFEST_FILENAME = "index_manifest.json"

# Chunk metadata is flat and all-string by construction (see Chunk.metadata in
# core/chunker.py); these are the record keys that are *not* metadata.
_NON_METADATA_KEYS = frozenset({"id", "text", "embed_text"})

# Chroma writes in one transaction per add(). Batching keeps peak memory flat
# and gives scripts/build_index.py something to report progress against.
_WRITE_BATCH = 200


def load_chunk_records(chunks_dir: Path) -> list[dict[str, Any]]:
    """Read ``chunks.jsonl`` into a list of dicts, one per line."""
    path = Path(chunks_dir) / "chunks.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing {path}. Run: python scripts/build_chunks.py")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _client(index_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(index_dir))


# What a Chroma persistence directory contains at the top level: the sqlite
# database, and one directory per vector segment named with a UUID.
_CHROMA_DB = "chroma.sqlite3"


def _clear(index_dir: Path) -> None:
    """Delete the whole index directory so a rebuild starts from nothing.

    ``client.delete_collection`` is not enough. It drops the collection from
    sqlite but leaves that collection's HNSW segment directory on disk, so
    every rebuild strands another one: measured at three directories after
    three builds. The index is committed to the repository, which turns each
    orphan into dead binary that git keeps forever.

    Removing the directory outright is destructive, so it only happens to a
    directory that actually holds an index. A mistyped ``INDEX_DIR`` pointing
    at real work must fail here rather than take that work with it.

    Deleting rather than reusing also avoids a subtler problem: Chroma keeps
    every segment's files open, so an orphan cannot be removed while a client
    on that directory is alive. Clearing before any client exists sidesteps
    the file locking entirely.
    """
    if not index_dir.exists():
        return
    if not index_dir.is_dir():
        raise RuntimeError(f"{index_dir} exists and is not a directory")

    entries = list(index_dir.iterdir())
    if not entries:
        return

    recognised = {_CHROMA_DB, MANIFEST_FILENAME}
    unexpected = [
        entry.name
        for entry in entries
        if entry.name not in recognised and not _looks_like_a_segment(entry)
    ]
    if unexpected:
        raise RuntimeError(
            f"refusing to rebuild into {index_dir}: it holds {len(unexpected)} "
            f"item(s) that are not part of a Chroma index. Point INDEX_DIR at "
            f"a directory this script owns, or empty this one by hand."
        )

    try:
        shutil.rmtree(index_dir)
    except PermissionError as exc:
        # Windows refuses to unlink a file another handle holds open. In
        # practice this means something in this process still has the index
        # open -- the build script opens it only after building, so the usual
        # cause is a Streamlit session or a REPL holding the old handle.
        raise RuntimeError(
            f"cannot clear {index_dir}: something still has the index open. "
            f"Close any running app or REPL using it and rebuild."
        ) from exc


def _looks_like_a_segment(entry: Path) -> bool:
    """True for a Chroma vector-segment directory, which is UUID-named."""
    if not entry.is_dir():
        return False
    try:
        uuid.UUID(entry.name)
    except ValueError:
        return False
    return True


def build(
    records: list[dict[str, Any]],
    embedder: Embedder,
    index_dir: Path,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Embed every record and write a fresh collection. Returns the count.

    The collection is dropped and recreated rather than updated in place. A
    full rebuild of this corpus costs about a third of a US cent, which is far
    less than the cost of reasoning about which chunks changed -- and an
    incremental index that quietly keeps a deleted chunk is a citation to a
    provision that no longer exists.

    ``progress`` is called with ``(done, total)`` after each batch, so the
    build script can show movement across a minute of paid API calls instead
    of sitting silent, where a hang looks exactly like slowness.
    """
    index_dir = Path(index_dir)
    _clear(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    client = _client(index_dir)

    # Closed in a finally: Chroma holds the sqlite file and every HNSW segment
    # open for the life of the client, and on Windows an open handle makes the
    # directory undeletable. Leaving it open would make the *next* rebuild in
    # the same process fail rather than this one.
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

        for start in range(0, len(records), _WRITE_BATCH):
            batch = records[start : start + _WRITE_BATCH]
            collection.add(
                ids=[record["id"] for record in batch],
                embeddings=embedder.embed([record["embed_text"] for record in batch]),
                documents=[record["text"] for record in batch],
                metadatas=[_metadata(record) for record in batch],
            )
            if progress is not None:
                progress(min(start + _WRITE_BATCH, len(records)), len(records))

        return collection.count()
    finally:
        client.close()


def _writable_copy(index_dir: Path, scratch_dir: Path | None) -> Path:
    """Copy the index somewhere writable and return where the copy landed.

    Chroma's persistent client is a read-write database, not a file reader:
    connecting opens a write transaction, and the first read flushes an HNSW
    buffer. Pointed straight at the committed ``data/index/`` that dirties
    files the repository tracks, and none of it converges -- part of
    ``length.bin`` is a buffer serialised with heap pointers still in it, which
    ASLR re-randomises every run, so there is no stable version that could be
    committed once to make it stop.

    It also makes the index unopenable on a read-only filesystem, which is a
    common container default and exactly the arrangement CLAUDE.md's "the app
    opens a prebuilt index read-only" describes.

    Copying outright is cheap enough that nothing cleverer is warranted:
    measured at 17 ms for the 13.6 MB corpus index. A scratch directory keyed
    on the manifest and reused between runs would save that and buy a cache
    invalidation rule plus a race between two processes populating it.
    """
    if scratch_dir is not None:
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="ai-act-index-", dir=scratch_dir))
    try:
        shutil.copytree(index_dir, workdir, dirs_exist_ok=True)
        # copytree preserves mode bits, so a read-only source would otherwise
        # produce a read-only copy -- the exact thing this function exists to
        # avoid. The copy belongs to this process, so it is ours to unlock.
        for path in (workdir, *workdir.rglob("*")):
            path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IRUSR)
    except OSError as exc:
        # The directory exists before the copy can be known to have worked, so
        # the failure path owns it. Leaving it to the atexit below would let a
        # long-lived process that retries a failing open strand one abandoned
        # copy per attempt.
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError(
            f"cannot open the index at {index_dir}: {exc}. The index is copied "
            f"to a writable scratch directory before it is opened, so this is a "
            f"permissions or free-space problem at one end or the other -- not "
            f"a missing or half-built index."
        ) from exc
    # The copy is this process's alone, so it dies with the process. 13.6 MB
    # left behind per run is not free on the kind of small host this deploys to.
    atexit.register(shutil.rmtree, workdir, ignore_errors=True)
    return workdir


def open_collection(index_dir: Path, scratch_dir: Path | None = None):
    """Open the prebuilt collection for reading, without writing to it.

    Never builds. CLAUDE.md requires the app to open a prebuilt index, so the
    only thing this can do about a missing one is say which script makes it.

    ``scratch_dir`` is where the writable copy goes; ``None`` means the
    system temp directory, which is what a deployed host normally wants.
    """
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise RuntimeError(
            f"no index at {index_dir}. Run: python scripts/build_index.py"
        )
    workdir = _writable_copy(index_dir, scratch_dir)
    try:
        return _client(workdir).get_collection(COLLECTION_NAME, embedding_function=None)
    except chromadb.errors.NotFoundError as exc:
        # Narrow on purpose. This message sends someone to rebuild, so it must
        # only appear when the collection really is absent -- the bare
        # `except Exception` it replaces reported a read-only filesystem as a
        # cancelled build, which is a wrong turn that costs an afternoon.
        raise RuntimeError(
            f"no '{COLLECTION_NAME}' collection in {index_dir} -- the directory "
            f"exists but holds no index, which is what a cancelled build leaves "
            f"behind. Run: python scripts/build_index.py"
        ) from exc


def _metadata(record: dict[str, Any]) -> dict[str, str]:
    """The record minus the fields Chroma stores separately.

    Derived by subtraction rather than by listing the metadata keys, so a
    field added to the chunker reaches the index without an edit here.
    """
    return {
        key: value for key, value in record.items() if key not in _NON_METADATA_KEYS
    }
