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

## Phase 2 — index and CLI retrieval (2026-09-19)

### Scope

- **Vector-only retrieval now; `rank_bm25` deferred** — fusing a lexical and a
  dense ranking needs a weight, and a weight chosen before there is a recall
  number to move is chosen by taste. `search()` returns a ranked list of
  scored candidates, which is the shape a fusion step consumes, so BM25 adds
  a function rather than rewriting `retrieve.py`.
- **The eval ships with retrieval, not after it** — choosing vector-first to
  establish a baseline only pays off if the baseline is measured, and
  CLAUDE.md's "report recall@5 before and after" is unenforceable until
  `evals/run_eval.py` exists.

### Deployment

- **`data/index/` is committed; the `.gitignore` entry was removed** —
  CLAUDE.md requires the app to open a prebuilt index read-only and forbids
  building at startup. A host that deploys straight from the repository has no
  build step and no persistent disk, so those two rules can only both hold if
  the index is in the repository. Measured at **12.9 MB**, well under
  GitHub's limits. Committing the index removes the build step, not the API
  key: the app still embeds each incoming question at runtime.
- **`scripts/build_index.py` writes `data/index/index_manifest.json`** —
  recording embed model, chunk count, build time and the sha256 of
  `chunks.jsonl`. A committed binary reveals nothing in a diff, so "are these
  vectors current for these chunks?" would otherwise be unanswerable by
  inspection. Mirrors `data/raw/manifest.json`.
- **A full rebuild every time, never incremental** — measured at ~170,600
  tokens, about **$0.0034** per rebuild. Reasoning about which chunks changed
  costs more than that, and an incremental index that quietly retains a
  deleted chunk is a citation to a provision that no longer exists.
- **`build()` deletes the whole index directory rather than calling
  `delete_collection`** — `delete_collection` drops the collection from
  sqlite but leaves its HNSW segment directory on disk. Measured: three
  rebuilds left three directories. With the index committed, each orphan is
  ~600 KB of dead binary that git keeps forever. The delete is guarded so it
  only ever removes a directory that actually holds an index; a mistyped
  `INDEX_DIR` pointing at real work raises instead.
- **`build()` closes its Chroma client in a `finally`** — Chroma holds the
  sqlite file and every segment open for the client's lifetime, and on
  Windows an open handle makes the directory undeletable. Leaving the client
  open does not break the build that opened it; it breaks the *next* one, so
  the symptom appears a step away from the cause. Measured directly: deleting
  an orphaned segment while a client is alive fails with `PermissionError`.

### Two Chroma defaults that fail silently

- **The collection is created with `configuration={"hnsw": {"space":
  "cosine"}}`** — Chroma 1.5.9 falls back to `l2` when no space is declared
  (`collection_configuration.py:442`). OpenAI embeddings are normalised for
  cosine; built in l2 the index still returns five results, just worse ones,
  and nothing raises.
- **Every `create_collection` and `get_collection` call passes
  `embedding_function=None`** — the default is `ONNXMiniLM_L6_V2`, and
  `onnxruntime` is already installed as a Chroma transitive. Left in place, a
  `query_texts=` call would embed the question with MiniLM and compare it
  against OpenAI document vectors — two unrelated spaces, no error — and would
  download an ~80 MB model on first use, a cold-start failure on a deployed
  host that never reproduces locally.
- **The resulting `{"type": "legacy"}` deprecation warning is filtered in the
  three tests that read the collection config, not globally** — verified that
  the build, open and query path never reads that config and is warning-free.
  A global filter would hide the warning if it started appearing somewhere new.

### Modules

- **`core/embed.py` exists although the documented layout does not name it** —
  indexing embeds 886 chunks offline and every query embeds one question at
  runtime, so putting the client in either `index.py` or `retrieve.py` would
  make the other depend on it for the wrong reason. `Embedder` is a Protocol
  and `OpenAIEmbedder` takes its client as a field, which is what lets the
  whole suite run with no network.
- **The OpenAI SDK is called directly rather than through `langchain-openai`**
  — one fewer layer between the reader and the HTTP call, and nothing in
  LangGraph requires LangChain's embedding wrapper. `langchain-openai` stays
  pinned for the generation phase.
- **No retry logic of our own** — the OpenAI client already retries transient
  failures, and a second layer would obscure which one caused a slow build.
- **Batched embedding sorts each response by the `index` the API reports** —
  the documented behaviour is that order is preserved, but a mismatch would
  file every vector under the wrong chunk and raise nothing. The sort is free.
- **Cosine distance is converted to a similarity once, in `search()`** —
  `score = 1 - distance`. Every surface above retrieval wants "higher is
  better"; converting in each caller means each caller gets the direction
  right by luck.
- **`Hit` is frozen** — it travels from retrieval through generation to the
  UI, and a stage that adjusted a score in passing would be very hard to find.
- **`build()` takes a `progress` callback** — the real build is nine paid API
  calls over about a minute, and in silence a hang is indistinguishable from
  slowness.
- **Chunk metadata is derived by subtracting `id`, `text` and `embed_text`
  from the record** — so a field added to the chunker reaches the index
  without an edit in `index.py`.

### Logging

- **`logs_dir` was added to `Config`; the log is `data/logs/queries.jsonl`,
  gitignored** — CLAUDE.md requires logging every query and requires all
  configuration to flow through `core/config.py`. The log is a local runtime
  artefact rather than a corpus artefact, and it carries user questions.
- **`rewritten` is written as `null` rather than omitted** — no rewrite node
  exists yet. Present-and-null keeps the log format stable; an absent key
  would mean every existing line needed special-casing once the graph lands.
- **Chunk text is not logged, only ids and scores** — the ids identify the
  text exactly, and a log that duplicates the corpus is one nobody reads.

### Evals

- **Gold is keyed on chunk id, not on the `article_no` label** — ids are the
  collection's primary key and match exactly. Matching labels would need
  prefix logic, and "Article 6" is a prefix of "Article 60".
- **`run_eval.py` verifies every gold id exists in the index before spending
  anything** — a typo in `expect` is indistinguishable from a retrieval
  failure once the numbers are printed. It just looks like a miss.
- **No threshold is enforced and a low score does not exit non-zero** — a
  quality bar encoded as a build failure is a bar people learn to route
  around. The obligation is to report the number, including when it worsens.
- **Misses print what came back instead, and flag gold found below the cut** —
  a gold chunk just outside the cut is a re-ranking or `k` problem; one absent
  entirely is an embedding problem. An aggregate number distinguishes neither.
- **MRR is truncated at `k` like recall is** — otherwise every row of a sweep
  prints the same number, since the rank of the first gold chunk does not
  depend on how many results were displayed.
- **The question set is split into `statutory` and `lay` groups, and the
  report breaks down by group** — the first 28 questions use the Act's own
  vocabulary, and because every chunk is embedded with its citation header
  attached they match on terminology almost for free: recall@5 over that
  subset alone saturated at **1.00**, leaving nothing for a later change to
  move. Eight questions asking the same subject matter in plain language were
  added for that reason, and the split is what makes the eval able to
  discriminate at all.

### Measured baseline — vector-only, `text-embedding-3-small`, top_k=5

Over 36 questions: **recall@5 0.83, recall@1 0.53, MRR@5 0.650.** By group:

| group | n | recall@5 | MRR@5 |
|---|---|---|---|
| statutory | 28 | 1.00 | 0.783 |
| lay | 8 | 0.25 | 0.188 |

Recall plateaus at 0.92 by k=10 and does not improve through k=20.

- **The dominant failure mode is recitals outranking articles on lay-phrased
  questions** — recitals are discursive prose and match conversational
  phrasing better than terse statutory text does, so they fill the top 5 while
  the operative Article sits below the cut. Four of the top five results for
  "can my company use AI to sift job applications" are recitals.

### Testing

- **`tests/test_config.py` holds regression guards, not test-driven design** —
  `core/config.py` already satisfied every assertion. Each pins a Phase 0
  decision that is cheap to undo by accident: a `.get()` with a default
  replacing `_require`, or a lost `field(repr=False)` printing the API key
  into the per-query log.
- **The fake embedder is a bag of words over a fixed vocabulary, not a hash**
  — hashed vectors are deterministic but arbitrary, so an assertion about
  which chunk ranks first would restate the hash rather than state anything
  about retrieval. With a bag of words, "which practices are prohibited"
  really is nearest the prohibitions chunk.
- **The suite was verified with all outbound sockets blocked and
  `OPENAI_API_KEY` unset** — 98 tests pass, which makes the no-network rule a
  measured fact rather than an intention.
## Phase 3 — generation (2026-09-21)

### Scope

- **Generation core only; the Streamlit app is its own step** — `core/generate.py`,
  `core/graph.py`, `scripts/ask.py` and their tests. Keeping the UI out holds
  this diff to something readable rather than skimmable, and the graph is
  exercised by a REPL in the meantime.
- **`core/graph.py` exists although the documented layout does not name it** —
  same precedent as `core/embed.py`. Prompts and model calls live in
  `generate.py`, wiring in `graph.py`, so every word the model sees is in one
  file and every edge is in the other. A single module mixing both would be
  ~350 lines with two unrelated reasons to change.
- **`evals/run_eval.py` was not re-run** — nothing in this phase calls into
  `core/retrieve.py`'s ranking and the file is untouched, so the recorded
  recall@5 of 0.83 still describes the index. The grounding check the commands
  block advertises is still unwritten and belongs with its own step.

### Refusal and citations

- **Refusal is model-judged and then verified, with no score threshold** — the
  prompt tells the model to refuse when the provisions do not answer, and every
  citation it emits is checked against what was actually retrieved. A minimum
  score would be a number chosen before any evidence says where it belongs,
  which is the objection that already deferred the BM25 fusion weight.
- **The refusal replaces the draft inside `answer()`, not in the caller** — a
  caller that forgot to check `refused` would otherwise print ungrounded prose
  about the law, and `generate.py` is the only module positioned to stop that.
- **`Answer.refused` is derived from `citations`, not stored** — so the flag
  cannot disagree with the evidence. An answer with nothing supporting it *is*
  a refusal; there is no third state.
- **Uncited prose is refused too** — which includes the model refusing in its
  own words. All three failure shapes end in one auditable refusal string.
- **Prose with inline bracketed citations, not structured output** — a JSON
  `citations` array detaches the citation from the sentence it supports, and
  CLAUDE.md's rule is that every *claim* cites. Brackets also give the app a
  place to hang a link to `source_url` later.
- **Citation matching is boundary-aware in both directions** — a cited label
  matches a retrieved one outright, or extends it at a `(` or `,` boundary, or
  is extended by it. Both directions are real: "Article 99" cited from the
  retrieved "Article 99(3)", and "Article 5(1)(a)" cited from "Article 5(1)".
  The boundary is what stops "Article 6" being satisfied by "Article 60(1)".
- **The narrowing direction additionally checks the chunk's text** — "Article
  5(1)(z)" is rejected because `(z)` appears nowhere in the Article 5(1) chunk,
  while `(a)` does. Without it, one genuine paragraph would vouch for any
  subdivision a model cared to invent under it.
- **Found by running it, not by reasoning about it** — the first real query was
  "Which AI practices are banned outright?", the single easiest question in the
  eval set, and it refused. The model had correctly cited Article 5(1)(a)-(h)
  from the Article 5(1) chunk, and the one-directional check called all ten
  fabrications. Phase 1 had already decided that citation precision does not
  require chunk precision; the check had simply not implemented it.
- **A refusal ships no sources** — `refuse_node` drops the hits. A refusal
  printed above five retrieved provisions reads, to anyone skimming, exactly
  like a cited answer.
- **`Answer.cited_hits` drives the sources list, not `hits`** — listing all
  `top_k` would credit an answer with provisions it never used.

### The graph

- **No checkpointer, by decision** — conversation history belongs to the
  caller, which keeps a turn a pure function of `(question, history)` and keeps
  it reproducible from the query log. Adding one would bring persistence,
  `thread_id` plumbing on every invoke, and state through serde, for a feature
  nothing has asked for.
- **Dependencies arrive as `build_graph()` arguments, never module globals** —
  what lets the tests build the same graph over a six-chunk index with a fake
  embedder and a canned generator, and never reach the network.
- **`log_query` moved into `retrieve_node`** — the CLI, the app and any future
  caller all log because none of them can retrieve without passing through the
  node. Logging in the CLI would have been one `--no-log` flag away from a
  silent gap.
- **`rewritten` is now populated** — the field `log_query` has written as
  `null` since Phase 2, exactly as its docstring anticipated. Question and
  rewrite are both recorded, which is the only way to tell a bad rewrite from
  a bad retrieval afterwards.

### Model

- **`chat_model` default moved from `gpt-4o-mini` to `gpt-5.6-terra`** — chosen
  from `client.models.list()` against the real account rather than typed from
  memory, the same reasoning as the dependency pins. Terra is the balanced tier
  at $2/$12 per million tokens, roughly $0.01 an answer at `top_k=5`. Luna is
  ten times cheaper and Sol 2.5 times dearer; both are one env var away.
- **`TEMPERATURE` default moved from 0.0 to 1.0** — measured: gpt-5.6 returns
  `400 unsupported_value` for every temperature but its default. Answers are
  held to the law by the citation check, not by a sampling parameter, and a
  `TEMPERATURE=0` left in an env file now fails loudly and by name.

### CLI

- **`scripts/ask.py` is a REPL, separate from `scripts/query.py`** — the
  documented layout pins `query.py` as retrieval with no generation, and
  keeping a generation-free view of retrieval is what makes a bad answer
  diagnosable. A loop rather than a one-shot because the rewrite node only does
  anything on a follow-up; a history passed by flags would have left that path
  unexercised until the UI existed.
- **Wrapping is per line, not over the whole answer** — `textwrap.wrap` on the
  full text collapses every newline, which turned a list of ten prohibitions
  into one unreadable paragraph.
- **Metadata prints in an explicit mid-gray, never ANSI dim** — dim renders as
  black on a black terminal.

### Testing

- **98 tests to 134, written before each module existed** — and verified with
  outbound connections blocked and `OPENAI_API_KEY` unset, so the no-network
  rule stays a measured fact.
- **`ExplodingGenerator` raises instead of counting calls** — where the
  contract is that no model call happens at all, a call counter would assert
  that a fake was used. Raising makes the real short-circuit failing show up as
  a test failure.
- **`FakeGenerator` returns canned completions** — an LLM's wording is not
  reproducible, but everything this project promises about an answer is: what
  went into the prompt, what came out, whether it refused.
- **The prompt-injection test asserts the system prompt is byte-identical** —
  cheap, because `SYSTEM_PROMPT` is a constant and corpus text only ever
  reaches `format_context`. The architecture is what makes the test trivial.
- **`tests/test_config.py`'s default assertions were updated deliberately** —
  they were red after the model and temperature change, which is the regression
  guard doing its job rather than a test to loosen.

## Phase 4 — lint and commit gate (2026-09-21)

### Lint configuration

- **The ruff rule set is pinned in the repository** — the rules that had been
  firing came from a user-level config outside the repo, so `ruff check`
  reported 26 findings here and almost none on a fresh clone. Ruff's own
  default is `E4`, `E7`, `E9` and `F`, which excludes `B017`, `UP031` and
  `RUF100` entirely. A gate can only enforce a standard the repository states.
- **That config lives in `ruff.toml`, not `pyproject.toml`** — uv decides a
  directory is a uv project from a pyproject's *existence*, not its contents.
  One added purely to hold `[tool.ruff]` made uv write a `uv.lock` declaring no
  dependencies, because this repo keeps them in `requirements.txt`. Measured:
  `uv sync --dry-run` against that lock listed **124** packages for removal,
  chromadb, langgraph and streamlit among them. A `ruff.toml` configures ruff
  identically and uv ignores it.
- **`RUF001`–`RUF003` ignored** — non-breaking spaces and typographic quotation
  marks are deliberate test data for a chunker whose whole job is surviving
  EUR-Lex markup. Enabling the rules would put a suppression comment on each of
  the 13 lines whose purpose is to hold the character.
- **`B905` ignored rather than fixed** — `zip(strict=True)` raises where the
  current code truncates, so it is a runtime behaviour change and not a lint
  fix. It needs a failing test written first.
- **Nineteen `# noqa: E402` directives removed** — each suppressed "module
  import not at top of file" on an import following a `sys.path.insert`. Ruff
  tolerates that idiom where flake8 does not, confirmed with `--ignore-noqa`,
  so all nineteen were dead suppressions from a linter this repo no longer uses.
- **Three `pytest.raises(Exception)` assertions narrowed to their real types** —
  a bare `Exception` also passes on an `AttributeError` from a renamed field, so
  those tests claimed to pin immutability while in fact reporting only that
  something went wrong. Each was confirmed to fail once its guarantee was
  removed. The Chroma one raises `ValueError`, established by running it.

### The commit gate

- **pre-commit installed only after the backlog was already clear** — a gate
  that is red on arrival is a gate people learn to bypass with `--no-verify`,
  which is the failure mode this file argues against for the eval threshold.
- **No `check-added-large-files`** — its 500 KB default would block
  `data/index/chroma.sqlite3`, the 13 MB file the no-build-step deploy depends
  on.
- **`trailing-whitespace` and `end-of-file-fixer` skip `data/`** —
  `index_manifest.json` pins a sha256 of `chunks.jsonl`, so a hook that
  appended a newline there would invalidate the manifest without touching the
  vectors it describes.
- **`ruff format` landed as its own commit** — it reformatted 15 files, and
  formatting sitting on top of a real change makes the real change unreadable.
  Every changed Python file was first verified to parse to an AST identical to
  the previous commit.
- **mypy and pytest stay out of it** — the suite is ~7 s, too slow to pay on
  every commit, and pre-commit's isolated environments have neither langgraph
  nor chromadb.

### Index churn

- **A hook refuses index files staged without `index_manifest.json`** — opening
  and querying the committed index makes Chroma rewrite headers and segment
  bookkeeping: measured at 372 bytes across ~13 MB after a handful of queries.
  Committing that writes a fresh 13 MB blob, because git stores binaries whole
  and cannot delta them, and the churn is indistinguishable from a real rebuild
  in `git status`. The manifest is what separates the two cases, since a
  rebuild rewrites it and a query does not.
- **The check asks `git diff --cached` rather than reading the paths pre-commit
  passes it** — pre-commit splits the file list across parallel invocations.
  Measured: one run handed a single invocation two of the six index files and
  no manifest, which as an argv-based check would reject a genuine rebuild.
  Asking git also yields the right semantics, because git lists only paths
  whose staged content differs from HEAD.
- **Churn is refused at commit time rather than avoided by copying the index on
  open** — a copy-on-open would put a 13 MB copy into every cold start and add
  a code path to `core/index.py` whose only purpose is protecting the
  repository. A repository problem is not worth solving in the running app.
