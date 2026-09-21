"""Structural contract for the chunker.

Every assertion here is a deterministic fact about parsing the real EUR-Lex
markup -- never a statement about retrieval quality, which belongs in
``evals/run_eval.py``. Each expected value was measured against the source
documents before this file was written.
"""

from __future__ import annotations

import re

import pytest

from core.chunker import (
    normalise,
    parse_consolidated,
    parse_oj_recitals,
    unrecognised_labels,
)

NBSP = " "


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_normalise_replaces_non_breaking_space_with_a_plain_space():
    assert normalise(f"Article{NBSP}75b") == "Article 75b"


def test_normalise_collapses_runs_of_whitespace():
    assert normalise("a  \n\t b") == "a b"


def test_normalise_drops_amendment_markers():
    # The consolidated text is littered with editorial markers showing which
    # amending act inserted or replaced a provision. They are not law.
    assert normalise("▼M1 The Commission shall") == "The Commission shall"
    assert normalise("►B The Commission shall") == "The Commission shall"


def test_normalise_drops_deletion_dashes():
    # Annex I, Section A carries "▼M1 —————" marking a repealed entry.
    assert normalise("▼M1 —————") == ""


# --------------------------------------------------------------------------
# The non-breaking space gotcha -- seven real headings carry it
# --------------------------------------------------------------------------


def test_article_heading_with_non_breaking_space_still_yields_its_number(fixture_html):
    # "Article\xa075b" -- one of seven headings (4, 4a, 60a, 75a-75d) that use a
    # non-breaking space. These are the provisions inserted by the 2026
    # amendment, so a naive split on " " silently corrupts the newest articles.
    chunks = parse_consolidated(fixture_html("article_75b_nbsp_heading.html"))
    assert {c.number for c in chunks} == {"75b"}


def test_annex_heading_with_non_breaking_space_still_yields_its_number(fixture_html):
    chunks = parse_consolidated(fixture_html("annex_14_nbsp_heading.html"))
    assert {c.number for c in chunks} == {"XIV"}


# --------------------------------------------------------------------------
# Articles: depth 2
# --------------------------------------------------------------------------


def test_article_splits_into_one_chunk_per_numbered_paragraph(fixture_html):
    chunks = parse_consolidated(fixture_html("article_06_paragraphs.html"))
    assert [c.paragraph for c in chunks] == [
        "1", "1a", "1b", "1c", "2", "3", "4", "5", "6", "7", "8",
    ]


def test_continuation_subparagraphs_join_the_paragraph_above_them(fixture_html):
    # Article 6(3) is followed by an unnumbered chapeau, four lettered points
    # and a closing subparagraph. All of it is paragraph 3 and must not become
    # five orphan chunks.
    chunks = parse_consolidated(fixture_html("article_06_paragraphs.html"))
    para_3 = next(c for c in chunks if c.paragraph == "3")
    assert "(a)" in para_3.text and "(d)" in para_3.text
    assert "Notwithstanding the first subparagraph" in para_3.text


def test_article_5_paragraph_1_keeps_its_chapeau_and_every_point(fixture_html):
    # The grounding test. Split point (a) away from "shall be prohibited" and
    # the chunk describes a practice in neutral terms, supporting an answer
    # that states the opposite of the law.
    chunks = parse_consolidated(fixture_html("article_05_prohibited.html"))
    para_1 = next(c for c in chunks if c.paragraph == "1")
    assert para_1.text.startswith("The following AI practices shall be prohibited:")
    assert len(re.findall(r"\((?:[a-z]|[a-z][a-z])\)", para_1.text)) >= 10
    assert "Point (h) of the first subparagraph" in para_1.text


def test_article_without_paragraphs_or_points_stays_whole(fixture_html):
    chunks = parse_consolidated(fixture_html("article_87_no_paragraphs.html"))
    assert len(chunks) == 1
    assert chunks[0].article_no == "Article 87"
    assert chunks[0].paragraph == ""


def test_article_without_paragraphs_splits_at_its_points(fixture_html):
    # Article 3 is a flat list of definitions with no paragraph level at all.
    chunks = parse_consolidated(fixture_html("article_03_definitions.html"))
    assert [c.article_no for c in chunks] == [f"Article 3({n})" for n in range(1, 7)]


def test_point_chunks_inherit_their_chapeau(fixture_html):
    chunks = parse_consolidated(fixture_html("article_03_definitions.html"))
    assert all(
        c.text.startswith(
            "For the purposes of this Regulation, the following definitions apply:"
        )
        for c in chunks
    )


def test_quoted_insertions_do_not_become_subdivisions(fixture_html):
    # Article 105 amends Directive 2014/90/EU by inserting a paragraph. The
    # inserted text is quoted and carries the *directive's* numbering, so its
    # marker reads "‘5." -- citing that as "Article 105(‘5)" is both malformed
    # and attributes the directive's numbering to the AI Act.
    chunks = parse_consolidated(fixture_html("article_105_lead_then_paragraph.html"))
    assert len(chunks) == 1
    assert chunks[0].article_no == "Article 105"
    assert chunks[0].paragraph == ""
    # The lead names which directive is amended; without it the chunk floats.
    assert "In Article 8 of Directive 2014/90/EU" in chunks[0].text
    assert "safety components" in chunks[0].text


def test_folded_markers_are_reported_rather_than_swallowed(fixture_html):
    # The narrow whitelist makes an unrecognised marker vanish into a
    # whole-article chunk. It has to be visible, or a new EUR-Lex drafting form
    # would quietly cost us every subdivision in that article.
    html = fixture_html("article_105_lead_then_paragraph.html")
    assert unrecognised_labels(html) == [("art_105", "‘5")]


def test_normal_articles_report_no_folded_markers(fixture_html):
    assert unrecognised_labels(fixture_html("article_06_paragraphs.html")) == []


def test_lettered_point_chunks_are_cited_by_their_letter(fixture_html):
    chunks = parse_consolidated(fixture_html("article_75b_nbsp_heading.html"))
    assert [c.article_no for c in chunks] == [
        "Article 75b(a)", "Article 75b(b)", "Article 75b(c)",
    ]


# --------------------------------------------------------------------------
# Annexes: split at sections
# --------------------------------------------------------------------------


def test_annex_with_sections_splits_at_them(fixture_html):
    chunks = parse_consolidated(fixture_html("annex_01_sections.html"))
    assert [c.article_no for c in chunks] == [
        "Annex I, Section A", "Annex I, Section B",
    ]


def test_annex_without_sections_stays_whole(fixture_html):
    chunks = parse_consolidated(fixture_html("annex_03_whole.html"))
    assert len(chunks) == 1
    assert chunks[0].article_no == "Annex III"


def test_annex_xi_pairs_its_split_section_headings(fixture_html):
    # Annex XI writes "Section 1" in one heading tag and the actual title in the
    # next. Unpaired, this yields two 9-character chunks that say nothing.
    chunks = parse_consolidated(fixture_html("annex_11_heading_pair.html"))
    assert len(chunks) == 2
    assert all(len(c.text) > 100 for c in chunks)
    assert "Information to be provided by all providers" in chunks[0].title


# --------------------------------------------------------------------------
# Recitals: the OJ document, a separate parser
# --------------------------------------------------------------------------


def test_recitals_are_one_chunk_each(fixture_html):
    chunks = parse_oj_recitals(fixture_html("recitals_oj.html"))
    assert [c.article_no for c in chunks] == ["Recital 1", "Recital 12", "Recital 53"]


def test_recital_number_is_not_welded_onto_its_text(fixture_html):
    # The number sits in its own table cell; a naive get_text() produces
    # "(12)\n\n\nThe notion of ...".
    chunks = parse_oj_recitals(fixture_html("recitals_oj.html"))
    recital_12 = next(c for c in chunks if c.number == "12")
    assert recital_12.text.startswith("The notion of")


def test_recitals_carry_no_chapter(fixture_html):
    chunks = parse_oj_recitals(fixture_html("recitals_oj.html"))
    assert all(c.chapter == "" for c in chunks)


def test_the_consolidated_parser_finds_no_recitals(fixture_html):
    # EUR-Lex drops recitals from consolidated versions. Two renditions, two
    # parsers: the class names (norm vs oj-normal) share nothing.
    assert parse_consolidated(fixture_html("recitals_oj.html")) == []


# --------------------------------------------------------------------------
# The metadata contract, across every fixture
# --------------------------------------------------------------------------

CONSOLIDATED_FIXTURES = [
    "article_06_paragraphs.html",
    "article_05_prohibited.html",
    "article_03_definitions.html",
    "article_87_no_paragraphs.html",
    "article_75b_nbsp_heading.html",
    "article_105_lead_then_paragraph.html",
    "annex_01_sections.html",
    "annex_03_whole.html",
    "annex_11_heading_pair.html",
    "annex_14_nbsp_heading.html",
]


@pytest.fixture(scope="session")
def every_chunk(fixture_html):
    chunks = []
    for name in CONSOLIDATED_FIXTURES:
        chunks.extend(parse_consolidated(fixture_html(name)))
    chunks.extend(parse_oj_recitals(fixture_html("recitals_oj.html")))
    return chunks


def test_every_chunk_carries_a_citation_label(every_chunk):
    # Chroma silently DROPS metadata keys whose value is None, so a nullable
    # article_no surfaces as a KeyError in generate.py at answer time rather
    # than as a failure at index time.
    assert every_chunk
    assert all(isinstance(c.article_no, str) and c.article_no for c in every_chunk)


def test_subdivision_labels_are_well_formed(every_chunk):
    # A malformed label means a citation nobody can look up. Quoted insertions
    # in the amendment articles are the way this goes wrong in practice.
    allowed = re.compile(r"|[0-9]+[a-z]*|[a-z]{1,2}|Section \S+")
    bad = [
        (c.id, c.paragraph)
        for c in every_chunk
        if not allowed.fullmatch(c.paragraph)
    ]
    assert bad == []


def test_every_chunk_has_non_empty_text(every_chunk):
    assert all(c.text.strip() for c in every_chunk)


def test_no_chunk_text_contains_an_amendment_marker(every_chunk):
    assert not [c.id for c in every_chunk if "▼" in c.text or "►" in c.text]


def test_ids_are_unique(every_chunk):
    ids = [c.id for c in every_chunk]
    assert len(ids) == len(set(ids))


def test_embed_text_leads_with_the_citation_label(every_chunk):
    assert all(c.embed_text.startswith(c.article_no) for c in every_chunk)


def test_embed_text_contains_the_full_text(every_chunk):
    assert all(c.text in c.embed_text for c in every_chunk)


def test_articles_resolve_their_chapter(fixture_html):
    chunks = parse_consolidated(fixture_html("article_06_paragraphs.html"))
    assert all(c.chapter == "Chapter III — HIGH-RISK AI SYSTEMS" for c in chunks)


def test_source_url_is_derived_from_the_documents_canonical_link(fixture_html):
    article = parse_consolidated(fixture_html("article_06_paragraphs.html"))[0]
    recital = parse_oj_recitals(fixture_html("recitals_oj.html"))[0]
    assert article.source_url == (
        "https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng#art_6"
    )
    assert recital.source_url == (
        "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng#rct_1"
    )


def test_chunks_are_serialisable_to_flat_scalar_metadata(every_chunk):
    # Whatever Chroma stores must be flat scalars; a None or a nested value
    # either raises or is silently dropped.
    for chunk in every_chunk:
        for key, value in chunk.metadata().items():
            assert isinstance(value, str), f"{chunk.id}.{key} is {type(value)}"
