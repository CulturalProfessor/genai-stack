"""The conversation list in the sidebar, shared by the chat pages.

Both the code chat and the PDF chat are the same interaction over the same
store, differing only in `kind` and in whether they are scoped to a subject
(a PDF). Keeping the widget here means one behaviour to fix, not two.
"""
from __future__ import annotations

import streamlit as st

from genai_stack.conversations import (
    UNTITLED,
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    rename_conversation,
)

# The selected conversation is per kind and per subject: switching to the PDF
# page must not drag the code chat's selection along with it.
def _selection_key(kind: str, subject: str | None) -> str:
    return f"conversation:{kind}:{subject or ''}"


def _renaming_key(kind: str, subject: str | None) -> str:
    return f"renaming:{kind}:{subject or ''}"


def current_conversation_id(graph, kind: str, subject: str | None = None) -> str:
    """The open conversation, creating one on first visit.

    Also recovers when the selected conversation no longer exists - deleted in
    another tab, or wiped from the database underneath the session.
    """
    key = _selection_key(kind, subject)
    conversation_id = st.session_state.get(key)
    if conversation_id and get_conversation(graph, conversation_id):
        return conversation_id

    existing = list_conversations(graph, kind=kind, subject=subject)
    conversation_id = (
        existing[0].id
        if existing
        else create_conversation(graph, kind=kind, subject=subject)
    )
    st.session_state[key] = conversation_id
    return conversation_id


def start_new(graph, kind: str, subject: str | None = None) -> str:
    """Open a fresh chat, reusing an empty one rather than piling them up."""
    for conversation in list_conversations(graph, kind=kind, subject=subject):
        if conversation.message_count == 0:
            st.session_state[_selection_key(kind, subject)] = conversation.id
            return conversation.id

    conversation_id = create_conversation(graph, kind=kind, subject=subject)
    st.session_state[_selection_key(kind, subject)] = conversation_id
    return conversation_id


def _rename_form(graph, conversation, key: str, renaming_key: str) -> None:
    with st.form(f"rename-{key}", clear_on_submit=True):
        title = st.text_input(
            "Title", value=conversation.title, label_visibility="collapsed"
        )
        save, cancel = st.columns(2)
        if save.form_submit_button("Save", use_container_width=True):
            rename_conversation(graph, conversation.id, title)
            st.session_state[renaming_key] = None
            st.rerun()
        if cancel.form_submit_button("Cancel", use_container_width=True):
            st.session_state[renaming_key] = None
            st.rerun()


def conversation_sidebar(
    graph, kind: str, subject: str | None = None, header: str = "Conversations"
) -> None:
    selection_key = _selection_key(kind, subject)
    renaming_key = _renaming_key(kind, subject)
    active = st.session_state.get(selection_key)

    with st.sidebar:
        st.markdown(f"#### {header}")

        if st.button(
            "New chat", key=f"new-{kind}-{subject}", use_container_width=True,
            type="primary",
        ):
            start_new(graph, kind, subject)
            st.session_state[renaming_key] = None
            st.rerun()

        conversations = list_conversations(graph, kind=kind, subject=subject)
        if not conversations:
            st.caption("No conversations yet.")
            return

        for conversation in conversations:
            key = conversation.id
            if st.session_state.get(renaming_key) == key:
                _rename_form(graph, conversation, key, renaming_key)
                continue

            open_col, menu_col = st.columns([5, 1])
            label = conversation.title
            if conversation.title == UNTITLED:
                label = f"_{UNTITLED}_"
            open_col.button(
                label,
                key=f"open-{key}",
                use_container_width=True,
                # The open chat is the filled one, so the list shows where you are.
                type="secondary" if key != active else "primary",
                on_click=lambda cid=key: st.session_state.update({selection_key: cid}),
            )
            with menu_col.popover("⋯", use_container_width=True):
                st.caption(f"{conversation.message_count} messages")
                if st.button("Rename", key=f"rename-{key}", use_container_width=True):
                    st.session_state[renaming_key] = key
                    st.rerun()
                if st.button("Delete", key=f"delete-{key}", use_container_width=True):
                    delete_conversation(graph, key)
                    if st.session_state.get(selection_key) == key:
                        st.session_state[selection_key] = None
                    st.rerun()
