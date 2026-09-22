# Test fixtures

Real EUR-Lex markup, sliced out of the documents recorded in
`data/raw/manifest.json`. The chunker's whole job is surviving the actual
markup, so these are genuine slices rather than hand-written HTML.

`data/raw/` is gitignored, so the suite reads from here instead: a test that
depended on the fetched corpus would not run on a fresh clone.

Each file is wrapped in a minimal `<html>` with the source document's own
`<link rel="canonical">`, because chunks derive `source_url` from it rather than
from a hardcoded string.

Articles are wrapped in a copy of their chapter (`div#cpt_*`) with sibling
articles removed, so chapter resolution is exercised against the real nesting —
including the `cpt_III.sct_1` section divs that share the `cpt_` id prefix.

| file | source id | why it exists |
|---|---|---|
| `article_06_paragraphs.html` | `art_6` | 11 numbered paragraphs; paragraph 3 carries a chapeau, four lettered points and a closing subparagraph as continuation content |
| `article_05_prohibited.html` | `art_5` | the grounding case — paragraph 1 is a chapeau plus ten prohibitions, and a trailing subparagraph that belongs to it |
| `article_03_definitions.html` | `art_3` | no paragraph level at all; splits at definition points. **Trimmed to the first 6 of 68** definitions, which are pure repetition |
| `article_87_no_paragraphs.html` | `art_87` | neither paragraphs nor points; stays whole |
| `article_75b_nbsp_heading.html` | `art_75b` | heading is `Article 75b` — one of seven headings using a non-breaking space, all inserted by the 2026 amendment |
| `article_105_lead_then_paragraph.html` | `art_105` | amends another directive by quoting the inserted text, so its marker is `‘5.` and belongs to Directive 2014/90/EU, not to Article 105 |
| `annex_01_sections.html` | `anx_I` | two `Section A.`/`Section B.` headings; Section A opens with a `▼M1 —————` repealed entry |
| `annex_03_areas.html` | `anx_III` | no section headings and long enough to split at its numbered areas. **Untrimmed**, because the eight areas are eight distinct domains rather than repetition, and each is now its own chunk |
| `annex_09_short_points.html` | `anx_IX` | no section headings and numbered points, but only 719 characters, so it stays whole — the counter-case that keeps the split from reaching every annex |
| `annex_11_heading_pair.html` | `anx_XI` | writes `Section 1` in one heading tag and the real title in the next; unpaired this yields two 9-character chunks |
| `annex_14_nbsp_heading.html` | `anx_XIV` | heading is `ANNEX XIV`, and it has no `title-annex-2` subtitle |
| `recitals_oj.html` | `rct_1`, `rct_12`, `rct_53` | from the **as-published** document, whose class names share nothing with the consolidated text; the recital number sits in its own table cell |

To regenerate after a re-fetch, re-slice the same ids from the files named in
the manifest. If a slice no longer reproduces the structure its row describes,
that is a finding about EUR-Lex, not a reason to edit the expectation.
