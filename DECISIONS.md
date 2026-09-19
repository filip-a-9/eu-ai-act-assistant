# Decisions

One line per decision, plus why. Newest phase at the bottom.

## Phase 0 — skeleton (2026-09-19)

### Environment

- **Python 3.12, not the 3.11 named in the original stack** — 3.11 was not
  installed and nothing in the stack requires it; the whole dependency set
  resolved on 3.12 with no source builds.
- **Virtualenv at `.venv/`, gitignored** — standard, and keeps the ~130
  transitive packages out of the repo.

### Dependencies

- **Direct dependencies pinned with `==`, transitives left to the resolver** —
  CLAUDE.md requires `==` pins; pinning all ~130 transitives would make
  `requirements.txt` unreadable, and readability was the stated priority. Run
  `pip freeze` if a deploy ever needs a full lock.
- **Versions resolved by installing unpinned and freezing, not guessed** — a pin
  that has never been installed is a guess, and guesses fail on someone else's
  machine, not yours.

### Config

- **stdlib frozen dataclass + `python-dotenv`, not pydantic-settings** — every
  line is readable top to bottom with no implicit field-name-to-env-var
  derivation; the goal is a repo the author fully understands.
- **`openai_api_key` carries `field(repr=False)`** — CLAUDE.md requires logging
  every query, and a config object that prints its own API key into those logs
  is a leak waiting to happen. Verified: the key is absent from `repr()` but
  present on the object.
- **Paths anchor to `PROJECT_ROOT`, not the working directory** — so
  `python scripts/fetch_source.py` behaves identically from any directory.
- **Separate `raw_dir()` accessor alongside `load_config()`** — downloading
  public EUR-Lex HTML needs no OpenAI key, and requiring one to run the first
  script on a fresh clone would be a pointless barrier.
- **Missing or malformed env vars raise `RuntimeError` naming the variable** —
  not `KeyError`, not a silent `None` that surfaces as a confusing failure three
  layers deeper.

### Corpus source

- **Fetch two documents, not one** — Regulation (EU) 2026/1744 ("Digital
  Omnibus on AI", in force 2026-07-08) amended the AI Act, so the as-published
  2024 text is superseded. Article 5 alone grew from 11,230 to 13,383
  characters between the two.
- **Articles and annexes will come from the consolidated text
  (`02024R1689-20260727`)** — it is the law actually in force. A compliance
  assistant confidently quoting repealed wording is a worse outcome than one
  that refuses.
- **Recitals will come from the as-published OJ text (`32024R1689`)** — EUR-Lex
  drops recitals from consolidated versions (measured: 180 vs 0), and recitals
  are where the Act explains its own intent.
- **HTML, not Formex/Akoma Ntoso XML** — the EU Cellar API returned HTTP 400 on
  plain content negotiation, and the HTML already carries stable structural
  anchors (`id="art_N"`, `id="rct_N"`, `id="anx_N"`), so XML would add work and
  a dependency for no gain.
- **The fetch script downloads only; it never parses** — keeps acquisition
  reproducible and independent of parser changes, and CLAUDE.md forbids network
  access at app startup.
- **`manifest.json` records url, CELEX, UTC timestamp, HTTP status, byte count
  and sha256 per file** — provenance for a legal corpus is not optional; this is
  how anyone verifies their local copy matches the one an index was built from.
- **Fetch is idempotent, re-downloading only on sha256 mismatch or `--force`** —
  safe to re-run while developing.
- **Explicit `User-Agent`** — EUR-Lex is unfriendly to the default
  `python-requests` agent.

### Repository

- **`data/raw/*` is gitignored but `data/raw/manifest.json` is committed** — the
  2 MB corpus is reproducible from the script; the checksums are not.
  Note the glob: `data/raw/` would stop git descending into the directory and
  the negation would silently never fire.
- **Claude Code tooling files are gitignored** — the published portfolio repo
  should read as ordinary project work.
- **No placeholder modules for later phases** — empty `chunker.py` /
  `retrieve.py` / `app.py` files are noise; each phase creates its own.
- **Added `scripts/fetch_source.py`, which the documented layout does not list** —
  the layout names only `build_index.py`, but acquisition and indexing are
  separate concerns with different failure modes and different reasons to re-run.