"""Turn the fetched EUR-Lex HTML into citable chunks.

Two renditions, two parsers. The consolidated text (articles and annexes) and
the as-published OJ text (recitals) share no CSS class names -- ``norm`` versus
``oj-normal`` -- so a single parser with a flag would rot. They stay separate.

Granularity is structural, never fixed-size:

* an article splits at its own numbered paragraphs;
* an article with no paragraph level splits at its lettered or numbered points;
* an article with neither stays whole;
* an annex splits at its Section headings, or stays whole if it has none;
* a recital is always one chunk.

Nothing here touches the network or reads ``data/raw`` directly; callers pass
HTML in.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

# Editorial markers showing which amending act inserted, replaced or deleted a
# provision (▼M1, ►B, ▼B). They are navigation aids, not law.
_MARKER = re.compile(r"[▼►][A-Z]\d*")
# A run of em dashes marks a repealed entry, e.g. Annex I: "▼M1 —————".
_DELETED = re.compile(r"—{2,}")
_WHITESPACE = re.compile(r"\s+")

# "Section A. List of ..." / "Section 1" / "1. Introduction" -- the three forms
# an annex subdivision heading takes across the fourteen annexes.
_SECTION_HEADING = re.compile(
    r"^(?:Section\s+)?([A-Z]|\d+)[.—-]?\s*(.*)$", re.IGNORECASE
)

# A real subdivision marker is "1.", "1a." or "(a)". The amendment articles
# (105-107, 109, 110) quote text they insert into *other* instruments, so their
# markers open with a quotation mark and carry the other instrument's
# numbering: "‘5.". Treating that as a subdivision of this Article produces the
# citation "Article 105(‘5)", which is malformed and attributes Directive
# 2014/90/EU's numbering to the AI Act. Anything not matching is not a
# subdivision, and the whole article stays in one chunk.
_SUBDIVISION_LABEL = re.compile(r"[0-9]+[a-z]*|[a-z]{1,2}")

_ARTICLE_SKIP = {"title-article-norm", "eli-title", "modref"}
_ANNEX_SKIP = {"title-annex-1", "title-annex-2", "separator-annex", "modref"}


@dataclass(frozen=True)
class Chunk:
    """One retrievable, citable unit of the Regulation."""

    id: str
    kind: str          # article | annex | recital
    article_no: str    # the rendered citation label, never empty
    number: str        # "6", "XIV", "27"
    paragraph: str     # "1a", "a", "Section A" -- "" where not applicable
    parent_id: str     # "" for whole units and recitals
    title: str
    chapter: str       # "" for recitals and annexes
    source_url: str
    text: str          # the law, clean
    embed_text: str    # citation header + text; what gets embedded

    def metadata(self) -> dict[str, str]:
        """The flat, all-string projection stored alongside the vector.

        Chroma silently drops keys whose value is ``None``, which would turn a
        missing citation into a ``KeyError`` at answer time rather than a
        failure at index time. Every value here is a string, possibly empty.
        """
        return {
            "kind": self.kind,
            "article_no": self.article_no,
            "number": self.number,
            "paragraph": self.paragraph,
            "parent_id": self.parent_id,
            "title": self.title,
            "chapter": self.chapter,
            "source_url": self.source_url,
        }


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Strip editorial apparatus and flatten whitespace.

    The non-breaking space matters more than it looks: seven article headings
    and one annex heading use U+00A0 instead of a space, and they are exactly
    the provisions inserted by the 2026 amendment.
    """
    text = text.replace(" ", " ")
    text = _MARKER.sub(" ", text)
    text = _DELETED.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _text_of(nodes: list[Tag]) -> str:
    """Join several sibling nodes into one normalised run of text."""
    return normalise(" ".join(n.get_text(" ") for n in nodes))


def _classes(node: Tag) -> set[str]:
    return set(node.get("class") or [])


def _soup(html: str) -> BeautifulSoup:
    """Parse with the HTML parser, deliberately.

    The as-published OJ file opens with an XML declaration, so BeautifulSoup
    warns that an XML parser would be more reliable. It would not be: both
    documents are served as ``text/html``, are addressed by HTML class names,
    and the OJ text's lettered points live in ``<table>`` markup that the HTML
    parser handles. The warning is suppressed here rather than at the call site
    so the reason travels with the decision.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        return BeautifulSoup(html, "lxml")


def _canonical(soup: BeautifulSoup) -> str:
    """The document's own ELI URL, so chunk links are never hardcoded."""
    link = soup.find("link", rel="canonical")
    return link["href"] if link and link.get("href") else ""


def _header(article_no: str, title: str, chapter: str) -> str:
    return " — ".join(p for p in (article_no, title, chapter) if p)


def _build(
    *,
    id: str,
    kind: str,
    article_no: str,
    number: str,
    paragraph: str,
    parent_id: str,
    title: str,
    chapter: str,
    source_url: str,
    text: str,
) -> Chunk:
    """Assemble a chunk, deriving ``embed_text`` from the citation header.

    ``text`` stays exactly as the law states it so the UI can quote it; the
    header lives only in ``embed_text``, which is what gets embedded and
    lexically indexed. A 96-character fragment is unretrievable without it.
    """
    header = _header(article_no, title, chapter)
    return Chunk(
        id=id,
        kind=kind,
        article_no=article_no,
        number=number,
        paragraph=paragraph,
        parent_id=parent_id,
        title=title,
        chapter=chapter,
        source_url=source_url,
        text=text,
        embed_text=f"{header}\n\n{text}",
    )


# ---------------------------------------------------------------------------
# Consolidated text: articles
# ---------------------------------------------------------------------------


def _chapter_of(node: Tag) -> str:
    """Resolve the chapter an article sits in.

    Section divs share the ``cpt_`` id prefix (``cpt_III.sct_1``), so they are
    excluded explicitly or every article in a sectioned chapter would report
    its section as its chapter.
    """
    chapter = node.find_parent(
        "div", id=lambda v: bool(v) and v.startswith("cpt_") and ".sct" not in v
    )
    if chapter is None:
        return ""
    heading = chapter.find("p", class_="title-division-1", recursive=False)
    name = chapter.find("p", class_="title-division-2", recursive=False)
    if heading is None:
        return ""
    roman = normalise(heading.get_text(" ")).replace("CHAPTER", "").strip()
    label = f"Chapter {roman}"
    return f"{label} — {normalise(name.get_text(' '))}" if name else label


def _paragraph_label(node: Tag) -> str:
    """Return "1a" for a numbered paragraph, or "" if this is not one."""
    if node.name != "div" or "norm" not in _classes(node):
        return ""
    marker = node.find("span", class_="no-parag")
    if marker is None:
        return ""
    label = normalise(marker.get_text(" ")).rstrip(".")
    return label if _SUBDIVISION_LABEL.fullmatch(label) else ""


def _point_label(node: Tag) -> str:
    """Return "a" or "1" for a lettered/numbered point, stripped of brackets."""
    column = node.find("div", class_="grid-list-column-1")
    if column is None:
        return ""
    label = normalise(column.get_text(" ")).strip("()").rstrip(".")
    return label if _SUBDIVISION_LABEL.fullmatch(label) else ""


def _article_chunks(article: Tag, canonical: str) -> list[Chunk]:
    art_id = article["id"]
    heading = article.find("p", class_="title-article-norm")
    subtitle = article.find("p", class_="stitle-article-norm")
    number = normalise(heading.get_text(" ")).replace("Article", "").strip() if heading else ""
    title = normalise(subtitle.get_text(" ")) if subtitle else ""
    chapter = _chapter_of(article)
    url = f"{canonical}#{art_id}"

    # Walk children in document order. A numbered paragraph opens a group;
    # everything after it belongs to that group until the next one. Article 5's
    # "Point (h) of the first subparagraph ..." and Article 6(3)'s chapeau plus
    # four lettered points are continuation subparagraphs, not orphans.
    groups: list[tuple[str, list[Tag]]] = []
    lead: list[Tag] = []
    points: list[tuple[str, Tag]] = []
    for child in article.find_all(recursive=False):
        classes = _classes(child)
        if classes & _ARTICLE_SKIP:
            continue
        label = _paragraph_label(child)
        if label:
            groups.append((label, [child]))
        elif groups:
            groups[-1][1].append(child)
        elif "grid-container" in classes and _point_label(child):
            points.append((_point_label(child), child))
        else:
            lead.append(child)

    chapeau = _text_of(lead)

    if groups:
        chunks = []
        for index, (label, nodes) in enumerate(groups):
            body = _text_of(nodes)
            # The paragraph number is already carried in `paragraph`; drop the
            # duplicate "1. " prefix so the text opens on the law itself.
            body = re.sub(rf"^{re.escape(label)}\.\s*", "", body)
            if index == 0 and chapeau:
                body = f"{chapeau} {body}"
            chunks.append(
                _build(
                    id=f"{art_id}.para_{label}",
                    kind="article",
                    article_no=f"Article {number}({label})",
                    number=number,
                    paragraph=label,
                    parent_id=art_id,
                    title=title,
                    chapter=chapter,
                    source_url=url,
                    text=body,
                )
            )
        return chunks

    if points:
        # No paragraph level at all -- Article 3's definitions, Article 75b's
        # commitments. The chapeau rides on every point: without "The following
        # AI practices shall be prohibited:" a point reads as a neutral
        # description of the practice it actually forbids.
        chunks = []
        for label, node in points:
            body = _text_of([node])
            chunks.append(
                _build(
                    id=f"{art_id}.point_{label}",
                    kind="article",
                    article_no=f"Article {number}({label})",
                    number=number,
                    paragraph=label,
                    parent_id=art_id,
                    title=title,
                    chapter=chapter,
                    source_url=url,
                    text=f"{chapeau} {body}".strip(),
                )
            )
        return chunks

    return [
        _build(
            id=art_id,
            kind="article",
            article_no=f"Article {number}",
            number=number,
            paragraph="",
            parent_id="",
            title=title,
            chapter=chapter,
            source_url=url,
            text=chapeau,
        )
    ]


# ---------------------------------------------------------------------------
# Consolidated text: annexes
# ---------------------------------------------------------------------------


def _section_boundaries(children: list[Tag]) -> list[tuple[int, int, str]]:
    """Find section headings, pairing the split ones.

    Annex XI writes ``Section 1`` in one heading tag and the actual title in
    the next. Unpaired, that yields two nine-character chunks that say nothing.
    Returns (heading_index, content_start_index, heading_text) per section.
    """
    indices = [
        i for i, c in enumerate(children)
        if "title-gr-seq-level-1" in _classes(c)
    ]
    boundaries: list[tuple[int, int, str]] = []
    skip = set()
    for position, index in enumerate(indices):
        if index in skip:
            continue
        text = normalise(children[index].get_text(" "))
        start = index + 1
        bare = re.fullmatch(r"(?:Section\s+)?([A-Z]|\d+)\.?", text, re.IGNORECASE)
        following = indices[position + 1] if position + 1 < len(indices) else None
        if bare and following == index + 1:
            text = f"{text} {normalise(children[following].get_text(' '))}"
            skip.add(following)
            start = following + 1
        boundaries.append((index, start, text))
    return boundaries


def _annex_chunks(annex: Tag, canonical: str) -> list[Chunk]:
    anx_id = annex["id"]
    heading = annex.find("p", class_="title-annex-1", recursive=False)
    subtitle = annex.find("p", class_="title-annex-2", recursive=False)
    number = normalise(heading.get_text(" ")).replace("ANNEX", "").strip() if heading else ""
    annex_title = normalise(subtitle.get_text(" ")) if subtitle else ""
    url = f"{canonical}#{anx_id}"

    children = [
        c for c in annex.find_all(recursive=False)
        if not (_classes(c) & _ANNEX_SKIP)
    ]
    boundaries = _section_boundaries(children)

    if not boundaries:
        return [
            _build(
                id=anx_id,
                kind="annex",
                article_no=f"Annex {number}",
                number=number,
                paragraph="",
                parent_id="",
                title=annex_title,
                chapter="",
                source_url=url,
                text=_text_of(children),
            )
        ]

    chapeau = _text_of(children[: boundaries[0][0]])
    chunks = []
    for ordinal, (_, start, text) in enumerate(boundaries, start=1):
        end = boundaries[ordinal][0] if ordinal < len(boundaries) else len(children)
        match = _SECTION_HEADING.match(text)
        label, section_title = match.groups() if match else (str(ordinal), text)
        body = _text_of(children[start:end])
        chunks.append(
            _build(
                id=f"{anx_id}.sec_{ordinal}",
                kind="annex",
                article_no=f"Annex {number}, Section {label}",
                number=number,
                paragraph=f"Section {label}",
                parent_id=anx_id,
                title=section_title or annex_title,
                chapter="",
                source_url=url,
                text=f"{chapeau} {body}".strip(),
            )
        )
    return chunks


def unrecognised_labels(html: str) -> list[tuple[str, str]]:
    """Subdivision markers the parser declined to split on.

    ``_SUBDIVISION_LABEL`` is a deliberately narrow whitelist, so whatever it
    rejects is folded silently into a whole-article chunk. That is the right
    outcome for the five known quoted insertions and the wrong one for a form
    EUR-Lex has not used yet, so the build asserts this list rather than
    discovering the difference in production.
    """
    soup = _soup(html)
    found = []
    for article in soup.select("div.eli-subdivision[id^=art_]"):
        for child in article.find_all(recursive=False):
            marker = (
                child.find("span", class_="no-parag")
                if "norm" in _classes(child)
                else child.find("div", class_="grid-list-column-1")
                if "grid-container" in _classes(child)
                else None
            )
            if marker is None:
                continue
            label = normalise(marker.get_text(" ")).strip("()").rstrip(".")
            if not _SUBDIVISION_LABEL.fullmatch(label):
                found.append((article["id"], label))
    return found


def parse_consolidated(html: str) -> list[Chunk]:
    """Chunk the consolidated text: articles and annexes, in document order."""
    soup = _soup(html)
    canonical = _canonical(soup)
    chunks: list[Chunk] = []
    for article in soup.select("div.eli-subdivision[id^=art_]"):
        chunks.extend(_article_chunks(article, canonical))
    for annex in soup.select("div[id^=anx_]"):
        chunks.extend(_annex_chunks(annex, canonical))
    return chunks


# ---------------------------------------------------------------------------
# As-published OJ text: recitals
# ---------------------------------------------------------------------------


def parse_oj_recitals(html: str) -> list[Chunk]:
    """Chunk the as-published text's recitals, one chunk each.

    Recitals come from this document because EUR-Lex drops them from
    consolidated versions. All 180 are structurally flat, so there is nothing
    below them to split on.
    """
    soup = _soup(html)
    canonical = _canonical(soup)
    chunks = []
    for recital in soup.select("[id^=rct_]"):
        number = recital["id"].removeprefix("rct_")
        # The number sits in its own table cell, so a naive extraction yields
        # "(12) The notion of ...". Drop the duplicated marker.
        text = re.sub(rf"^\({re.escape(number)}\)\s*", "", _text_of([recital]))
        chunks.append(
            _build(
                id=recital["id"],
                kind="recital",
                article_no=f"Recital {number}",
                number=number,
                paragraph="",
                parent_id="",
                title="",
                chapter="",
                source_url=f"{canonical}#{recital['id']}",
                text=text,
            )
        )
    return chunks
