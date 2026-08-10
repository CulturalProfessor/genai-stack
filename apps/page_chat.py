"""Ask questions about the indexed codebase, in saved conversations."""
from __future__ import annotations

import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.logger import get_logger

from genai_stack.chains import answer_question, generate_ticket
from genai_stack.conversations import (
    KIND_CODE,
    append_message,
    chat_history,
    load_messages,
    set_title_if_untitled,
)
from conversation_ui import conversation_sidebar, current_conversation_id
from resources import get_chains, get_graph, get_settings, llm_unavailable

logger = get_logger(__name__)

MODES = ["Graph RAG", "LLM only"]


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def replay(messages) -> None:
    """Every message of the conversation, oldest first."""
    for message in messages:
        with st.chat_message(message.role):
            if message.mode:
                st.caption(f"RAG: {message.mode}")
            st.write(message.content)


def ticket_expander(question: str, key: str) -> None:
    with st.expander("Turn this into a ticket"):
        st.caption("Drafts a well-formed issue from this question.")
        if not st.button("Draft ticket", key=f"ticket-{key}"):
            return
        try:
            title, body = generate_ticket(
                neo4j_graph=get_graph(),
                llm_chain=get_chains()[0],
                input_question=question,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.exception("Ticket generation failed")
            st.error(f"Error: {exc}", icon="🚨")
            return
        st.text_input("Title", title, key=f"ticket-title-{key}")
        st.text_area("Description", body, height=200, key=f"ticket-body-{key}")


def render() -> None:
    st.header("Ask the codebase")

    try:
        graph = get_graph()
    except Exception as exc:  # noqa: BLE001 - the page explains and stays up
        logger.exception("Could not reach Neo4j")
        st.error(f"Neo4j is not reachable: {exc}", icon="🚨")
        st.info("Conversations are stored in Neo4j, so the chat needs it.", icon="💡")
        return

    # Resolved before the sidebar so a first visit lists the chat it just made.
    conversation_id = current_conversation_id(graph, kind=KIND_CODE)
    conversation_sidebar(graph, kind=KIND_CODE)
    messages = load_messages(graph, conversation_id)

    mode = st.radio(
        "Retrieval", MODES, horizontal=True,
        help="Graph RAG searches the indexed code and follows its call and import edges. "
             "LLM only answers from the model's own knowledge.",
    )
    st.caption(f"Model: `{get_settings().llm}`")

    replay(messages)

    if messages and messages[-1].role == "assistant":
        # Only the newest turn - a ticket per historic message would be noise.
        last_question = next(
            (m.content for m in reversed(messages) if m.role == "user"), None
        )
        if last_question:
            ticket_expander(last_question, key=messages[-1].id)

    question = st.chat_input("What would you like to know about this codebase?")
    if not question:
        return

    try:
        llm_chain, rag_chain = get_chains()
    except Exception as exc:  # noqa: BLE001 - the page explains and stays up
        logger.exception("Could not build the chains")
        llm_unavailable(exc)
        return

    # Saved before the answer, so a question that fails is not lost from the
    # conversation - and titles the chat even if the model never replies.
    append_message(graph, conversation_id, "user", question)
    set_title_if_untitled(graph, conversation_id, question)

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.caption(f"RAG: {mode}")
        try:
            answer = answer_question(
                rag_chain if mode == "Graph RAG" else llm_chain,
                question,
                history=chat_history(messages),
                callbacks=[StreamHandler(st.empty())],
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.exception("Query failed")
            llm_unavailable(exc)
            return

    append_message(graph, conversation_id, "assistant", answer, mode=mode)
    # Rerun so the sidebar picks up the new title and the ticket drafter
    # attaches to the turn that just finished.
    st.rerun()
