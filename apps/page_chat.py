"""Ask questions about the indexed codebase."""
from __future__ import annotations

import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.logger import get_logger

from genai_stack.chains import generate_ticket
from resources import get_chains, get_graph, get_settings, llm_unavailable

logger = get_logger(__name__)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def display_history() -> None:
    history = st.session_state.setdefault("history", [])
    # Only the last three exchanges, so the page stays readable.
    for entry in history[-3:]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.caption(f"RAG: {entry['mode']}")
            st.write(entry["answer"])


def render() -> None:
    st.header("Ask the codebase")

    rag_mode = st.radio(
        "Retrieval", ["Graph RAG", "LLM only"], horizontal=True,
        help="Graph RAG searches the indexed code and follows its call and import edges. "
             "LLM only answers from the model's own knowledge.",
    )
    st.caption(f"Model: `{get_settings().llm}`")

    display_history()

    user_input = st.chat_input("What would you like to know about this codebase?")
    if not user_input:
        return

    try:
        llm_chain, rag_chain = get_chains()
    except Exception as exc:  # noqa: BLE001 - the page explains and stays up
        logger.exception("Could not build the chains")
        llm_unavailable(exc)
        return

    chain = rag_chain if rag_mode == "Graph RAG" else llm_chain

    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.caption(f"RAG: {rag_mode}")
        try:
            answer = chain(
                {"question": user_input, "chat_history": []},
                callbacks=[StreamHandler(st.empty())],
            )["answer"]
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.exception("Query failed")
            llm_unavailable(exc)
            return

    st.session_state.history.append(
        {"question": user_input, "answer": answer, "mode": rag_mode}
    )

    with st.expander("Turn this into a ticket"):
        st.caption("Drafts a well-formed issue from your last question.")
        if st.button("Draft ticket"):
            try:
                title, body = generate_ticket(
                    neo4j_graph=get_graph(),
                    llm_chain=get_chains()[0],
                    input_question=user_input,
                )
                st.text_input("Title", title)
                st.text_area("Description", body, height=200)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.exception("Ticket generation failed")
                st.error(f"Error: {exc}", icon="🚨")
