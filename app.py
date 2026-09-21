"""The chat surface, and nothing else.

Every decision about what to retrieve, what to send the model and when to
refuse is behind ``core.graph.run_turn``. This file holds no prompt text, no
ranking and no retrieval: it renders what a turn produced, and it decides when
not to spend money. That is the whole of its job, and it is why the CLI, the
eval and this app cannot answer the same question differently.

Streamlit re-executes this module top to bottom on every interaction, which is
the one fact its shape follows from. The index, the two API clients and the
compiled graph are built once behind ``@st.cache_resource``; everything about
the conversation lives in ``st.session_state``; and nothing between them is
allowed to hold state in a module global.

No ``sys.path`` insert, unlike the entry points in ``scripts/``: Streamlit runs
this file from the repository root, so ``core`` is already importable.
"""

from __future__ import annotations

import streamlit as st

from core.config import load_config
from core.embed import openai_embedder
from core.generate import Answer, link_citations, openai_generator
from core.graph import build_graph, run_turn
from core.index import open_collection

DISCLAIMER = (
    "Not legal advice. Answers are generated from the text of Regulation (EU) "
    "2024/1689 and cite it; verify anything you act on against the Official "
    "Journal."
)

# The third one is deliberately outside the corpus. A refusal is the thing this
# assistant is actually built to do, and a visitor who only ever asks answerable
# questions never sees it.
EXAMPLES = (
    "Which AI practices are prohibited?",
    "What makes an AI system high-risk?",
    "What does the GDPR say about consent?",
)


@st.cache_resource(show_spinner="Opening the index…")
def _assistant():
    """Config, index, clients and the compiled graph, built once per process.

    Cached because Streamlit reruns this module on every keystroke of the chat
    box. Rebuilt each time, this would reopen the Chroma collection and
    construct two API clients per interaction.
    """
    config = load_config()
    graph = build_graph(
        collection=open_collection(config.index_dir, config.scratch_dir),
        embedder=openai_embedder(config.openai_api_key, config.embed_model),
        generator=openai_generator(
            config.openai_api_key, config.chat_model, config.temperature
        ),
        top_k=config.top_k,
        logs_dir=config.logs_dir,
    )
    return config, graph


def render_answer(result: Answer) -> None:
    """One answer: the prose with its citations linked, then its sources.

    ``cited_hits`` rather than ``hits``: listing everything retrieved would
    credit the answer with provisions it never used. A refusal has neither,
    because the graph empties them, so it renders as bare text -- which is the
    intended reading. Nothing supported it.
    """
    st.markdown(link_citations(result.text, result.hits))
    if result.cited_hits:
        sources = "\n".join(
            f"- [{_label(hit)}]({hit.source_url})" for hit in result.cited_hits
        )
        st.caption("**Sources**\n\n" + sources)


def _label(hit) -> str:
    """How a source reads: ``Article 5 — Prohibited AI practices``.

    Falls back to the bare citation label where a chunk has no distinct title,
    which is the case for every recital.
    """
    if hit.title and hit.title != hit.article_no:
        return f"{hit.article_no} — {hit.title}"
    return hit.article_no


st.set_page_config(page_title="EU AI Act Compliance Assistant", page_icon="📘")
st.title("EU AI Act Compliance Assistant")
st.caption(DISCLAIMER)

try:
    config, graph = _assistant()
except RuntimeError as exc:
    # A missing key and a missing index both raise RuntimeError naming the fix.
    # Shown as the message it is, rather than as a traceback on the page.
    st.error(str(exc))
    st.stop()

# Two separate counters on purpose. ``turns`` is the conversation, which the
# visitor may clear; ``asked`` is what has been spent, which they may not. A
# cap that a button resets is not a cap.
if "turns" not in st.session_state:
    st.session_state.turns = []
if "asked" not in st.session_state:
    st.session_state.asked = 0

turns: list[tuple[str, Answer]] = st.session_state.turns
at_cap = st.session_state.asked >= config.max_questions

status, reset = st.columns([4, 1], vertical_alignment="center")
status.caption(f"{st.session_state.asked} of {config.max_questions} questions")
if reset.button("New conversation", disabled=not turns):
    st.session_state.turns = []
    st.rerun()

# Only before the first question: once there is a conversation, the starters
# are clutter, and clicking one mid-thread would read as a follow-up.
asked_by_click = None
if not turns:
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        if column.button(example, disabled=at_cap, width="stretch"):
            asked_by_click = example

for question, result in turns:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render_answer(result)

if at_cap:
    st.info(
        f"This demo answers {config.max_questions} questions per session, "
        "because each one calls a paid model. Reload the page to start over."
    )

typed = st.chat_input("Ask about the AI Act", disabled=at_cap)
question = typed or asked_by_click

if question and not at_cap:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Reading the Act…"):
        try:
            # History is the current thread only, in the shape the CLI passes:
            # one turn is a function of the question and what came before it.
            result = run_turn(graph, question, [(q, a.text) for q, a in turns])
        except Exception as exc:
            # Anything the API can raise -- rate limit, timeout, revoked key --
            # becomes a message in the thread rather than a blank page. The
            # turn is not recorded and is not counted, because it produced no
            # answer and, for most of these, no billable call.
            result = None
            st.error(f"That question could not be answered just now: {exc}")

    if result is not None:
        st.session_state.turns.append((question, result))
        st.session_state.asked += 1
        # Rerun so the counter, the cap and the starters all reflect the turn
        # that just landed; the answer is already in session_state, so this
        # repaints from memory rather than asking anything again.
        st.rerun()
