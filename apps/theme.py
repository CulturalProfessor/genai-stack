"""The look of the app: reading width, chat bubbles, sidebar rows.

Colours are theme options in `.streamlit/config.toml`. What is here is the
layout that Streamlit has no setting for - a chat transcript wants a column you
can read across, not the full width of a monitor, and a conversation list wants
rows that behave like rows rather than buttons.

Selectors are Streamlit's stable `data-testid` hooks plus the `st-key-*` class
that `st.container(key=...)` emits, so a restyle does not depend on the hashed
emotion class names that change between releases.
"""
from __future__ import annotations

import streamlit as st

# Wide enough for code blocks in an answer, narrow enough that prose does not
# run away from the eye. Roughly the measure every chat product settles on.
CHAT_WIDTH = "48rem"
PAGE_WIDTH = "62rem"

_CSS = """
<style>
/* ---------- shell ---------- */
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] {
    max-width: %(width)s;
    padding-top: 2.5rem;
    padding-bottom: 7rem;
}
[data-testid="stHeader"] { background: transparent; }

/* ---------- chat transcript ---------- */
[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0;
    margin-bottom: 1.6rem;
    gap: 0;
}
/* The avatars carry the role, which the bubble styling already says. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none; }

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    flex: 0 1 auto;
    max-width: 85%%;
    margin-left: auto;
    padding: 0.65rem 1rem;
    border-radius: 1.15rem 1.15rem 0.3rem 1.15rem;
    background: var(--secondary-background-color, #171A21);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    line-height: 1.72;
}
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

/* ---------- composer ---------- */
[data-testid="stBottomBlockContainer"] {
    max-width: %(width)s;
    padding-bottom: 1.1rem;
}
[data-testid="stChatInput"] {
    border-radius: 1.4rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
[data-testid="stChatInputTextArea"] { padding: 0.55rem 0.25rem; }

/* ---------- sidebar rows ---------- */
[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    text-align: left;
    font-weight: 450;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
}
[data-testid="stSidebar"] .stButton > button p {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    background: transparent;
    border-color: transparent;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.05);
}
/* The open conversation reads as selected, not as a call to action. */
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
    background: rgba(124, 156, 255, 0.14);
    border-color: transparent;
    color: var(--text-color, #E7E9EE);
}
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
    background: rgba(124, 156, 255, 0.22);
    color: var(--text-color, #E7E9EE);
}
/* ...except the one button that is a call to action. The sidebar prefix is
   needed to outweigh the row styling above, which is the more specific
   selector on its own. */
[data-testid="stSidebar"] .st-key-new-chat [data-testid="stBaseButton-primary"] {
    background: var(--primary-color, #7C9CFF);
    color: #0B0D12;
    font-weight: 600;
    justify-content: center;
    margin-bottom: 0.4rem;
}
[data-testid="stSidebar"] .st-key-new-chat [data-testid="stBaseButton-primary"]:hover {
    background: #94AEFF;
    color: #0B0D12;
}
/* Row actions stay out of the way until you reach for them. */
[data-testid="stSidebar"] [data-testid="stPopoverButton"] {
    opacity: 0;
    background: transparent;
    border-color: transparent;
    transition: opacity 120ms ease-in-out;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover
[data-testid="stPopoverButton"],
[data-testid="stSidebar"] [data-testid="stPopoverButton"]:focus { opacity: 1; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.15rem; }

/* ---------- sidebar section labels ---------- */
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.45;
    margin: 1rem 0 0.35rem 0.2rem;
}

/* ---------- retrieval switch (a radio, dressed as a segmented control) ---------- */
.st-key-retrieval [role="radiogroup"] {
    display: flex;
    gap: 0.2rem;
    padding: 0.2rem;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 0.6rem;
}
.st-key-retrieval label[data-baseweb="radio"] {
    flex: 1;
    margin: 0;
    justify-content: center;
    padding: 0.3rem 0.4rem;
    border-radius: 0.45rem;
    cursor: pointer;
    transition: background 120ms ease-in-out;
}
/* The dot itself; the selected pill already says which one is on. */
.st-key-retrieval label[data-baseweb="radio"] > div:first-child { display: none; }
.st-key-retrieval label[data-baseweb="radio"] p {
    font-size: 0.8rem;
    margin: 0;
}
.st-key-retrieval label[data-baseweb="radio"]:hover {
    background: rgba(255, 255, 255, 0.06);
}
.st-key-retrieval label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(124, 156, 255, 0.18);
}

/* ---------- starter prompts ---------- */
.st-key-starters [data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 1px solid var(--border-color, #242832);
    justify-content: flex-start;
    text-align: left;
    height: auto;
    min-height: 3rem;
    padding: 0.7rem 0.9rem;
    font-weight: 450;
    opacity: 0.85;
}
.st-key-starters [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(124, 156, 255, 0.45);
    opacity: 1;
}
.st-key-starters [data-testid="stBaseButton-secondary"] p {
    white-space: normal;
    line-height: 1.4;
}

/* ---------- empty state ---------- */
.empty-state {
    text-align: center;
    margin: 4.5rem 0 2rem;
}
.empty-state h2 {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.empty-state p {
    opacity: 0.55;
    font-size: 0.95rem;
    margin: 0;
}
</style>
"""


def apply(width: str = CHAT_WIDTH) -> None:
    """Inject the stylesheet. Cheap enough to run on every rerun."""
    st.html(_CSS % {"width": width})


def sidebar_label(text: str) -> None:
    st.markdown(f'<div class="sidebar-label">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="empty-state"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )
