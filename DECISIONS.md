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

## Phase 1 — chunking (2026-09-19)

### Granularity

- **An article splits at its own numbered paragraphs, not at the article
  boundary** — CLAUDE.md says to chunk on article / annex / recital boundaries,
  but Article 3 is 17,611 characters (68 definitions in one vector), Article 5
  is 13,267, and 40 articles exceed 3,000. At `top_k=5` that hands the model
  ~50k characters. The paragraph is still a structural boundary, just a finer
  one, so the rule bends without becoming fixed-size chunking.
- **Recursion stops at the paragraph; it does not descend to lettered points** —
  Article 5(1) stays whole at 5,406 characters, holding the chapeau and all ten
  prohibitions, because that is what a "what is banned?" question needs.
  Descending one level further would fragment the most-queried provision in the
  Act into ten rows competing for the same `top_k` budget, and 39% of the
  corpus would fall under 200 characters.
- **Citation precision does not require chunk precision** — a paragraph chunk
  keeps its `(a)`/`(b)` labels in the text, so an answer can still cite
  Article 5(1)(a) from it. That removes the main argument for splitting further.
- **An article with no paragraph level splits at its points instead** —
  Articles 3, 16, 66, 75b, 108, 110, 113. Without this, Article 3 alone is 5% of
  the corpus in a single vector.
- **Continuation subparagraphs join the paragraph above them** — Article 5(1) is
  followed by "Point (h) of the first subparagraph is without prejudice to
  Article 9…", and Article 6(3) by a chapeau, four lettered points and a closing
  subparagraph. These are parts of the paragraph they follow; selecting elements
  by class rather than walking children in document order turns them into
  orphans that cite the wrong paragraph.
- **Chapeau inheritance is mandatory below the paragraph level** — split
  Article 5(1)(a) from "The following AI practices shall be prohibited:" and the
  chunk reads as a neutral description of a practice, with nothing marking it
  prohibited. That chunk supports an answer stating the opposite of the law.
- **Annexes split at their Section headings, or stay whole** — a Section is to
  an annex what a paragraph is to an article. Splitting at numbered items
  instead would shatter Annex II into 16 items averaging 55 characters, each a
  bare criminal-offence name with no legal content.
- **A recital is always one chunk** — all 180 are structurally flat.

Result: 675 article + 31 annex + 180 recital = 886 chunks, median 432
characters, largest 7,291 (Annex III).

### Metadata and citations

- **`article_no` holds a rendered citation label and is never empty** —
  "Article 5(1)", "Annex I, Section A", "Recital 27". Measured: Chroma 1.5.9
  silently *drops* metadata keys whose value is `None`, so a nullable citation
  field would surface as a `KeyError` in `generate.py` at answer time instead of
  failing at index time.
- **Structured companions sit alongside the label** — `kind`, `number`,
  `paragraph`, `parent_id`. The label is derivable from them; the redundancy
  buys one field every consumer can render without reimplementing the format.
- **`parent_id` is stored now although nothing reads it yet** — it makes
  parent-document expansion a retrieval-phase decision that can be flipped and
  re-evaluated without rebuilding the index.
- **`text` is clean law; `embed_text` is citation header + text** — the header
  is what makes a 77-character chunk retrievable, but the UI must quote the law
  without our editorial prefix welded on. Both are committed, so review shows
  exactly what was embedded.
- **`source_url` derives from each document's own `<link rel="canonical">`** —
  the URL comes from the fetched artefact rather than a hardcoded string, and
  the ELI anchors (`#art_6`, `#rct_27`) already exist in the markup.
- **Quoted insertions do not become subdivisions** — Articles 105-107, 109 and
  110 amend other instruments by quoting the text they insert, so their markers
  carry the other instrument's numbering (`‘5.`). Left alone this produced the
  citation "Article 105(‘5)", attributing Directive 2014/90/EU's numbering to
  the AI Act. Those articles now stay whole, and `unrecognised_labels()` reports
  the five known cases so the build fails if a sixth form appears.

### Parsing

- **Two parsers, not one with a flag** — the consolidated and as-published texts
  share no class names (`norm` versus `oj-normal`), so a single parser
  parameterised by rendition would rot.
- **Non-breaking spaces are normalised before anything else** — seven article
  headings (`Article 4`, `4a`, `60a`, `75a`-`75d`) and `ANNEX XIV` use U+00A0
  instead of a space. Those are exactly the provisions inserted by the 2026
  amendment, so a naive split on `" "` corrupts the newest law and nothing else.
- **Amendment markers and repealed entries are stripped** — `▼M1`, `►B` and runs
  of em dashes are editorial apparatus, not law. Annex I Section A consequently
  starts at item 2, because item 1 is struck.
- **HTML parser, not XML, for both documents** — the OJ file opens with an XML
  declaration and BeautifulSoup warns about it, but both files are served as
  `text/html` and are addressed by HTML class names. The warning is suppressed
  at the single place that parses, with the reason attached.

### Testing

- **The suite was written before the chunker existed and reads fixtures, never
  `data/raw/`** — `data/raw/` is gitignored, so a suite depending on it would
  not run on a fresh clone. `tests/fixtures/` holds real slices, one per
  structural form, documented in `tests/fixtures/README.md`.
- **Writing the tests first found a real bug** — the `Article 105(‘5)` citation
  was produced by code that already passed 29 other tests. Every gotcha in this
  corpus fails silently rather than raising.
- **The whole-corpus count is asserted in `scripts/build_chunks.py`, not in
  pytest** — asserting 886 needs the full 2 MB corpus, and the test suite must
  run without it. The script also fails on duplicate ids, empty text, missing
  citations, retained markers and oversized embeddings.
- **`data/chunks/chunks.jsonl` is committed** — a chunker change then shows up
  as a readable diff over the corpus instead of an invisible change to an
  artefact nobody can see.
- **Added `scripts/build_chunks.py`, which the documented layout does not list** —
  same reasoning as `fetch_source.py`: parsing and indexing have different
  failure modes and different reasons to re-run, and only indexing needs an API
  key.