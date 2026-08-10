import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.conversations import (
    UNTITLED,
    Conversation,
    Message,
    append_message,
    chat_history,
    create_conversation,
    list_conversations,
    load_messages,
    rename_conversation,
    title_from,
)


class FakeGraph:
    """Records queries instead of running them, and replays canned rows."""

    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []
        self.calls = []

    def query(self, cypher, params=None):
        self.calls.append((" ".join(cypher.split()), params or {}))
        return self.rows


def test_title_from_uses_the_question():
    assert title_from("What does syncDataToUser do?") == "What does syncDataToUser do?"


def test_title_from_collapses_whitespace():
    assert title_from("  what   does\nthis do  ") == "what does this do"


def test_title_from_falls_back_when_blank():
    assert title_from("   \n  ") == UNTITLED


def test_title_from_breaks_on_a_word_boundary():
    question = "explain how the retrieval chain expands the graph before it answers"
    title = title_from(question)
    assert len(title) <= 63
    assert title.endswith("...")
    # Cut between words, so no half word is left dangling before the ellipsis.
    assert question.startswith(title[:-3])
    assert title[-4] != " "


def test_title_from_cuts_a_long_unbroken_token():
    title = title_from("a" * 200)
    assert title == "a" * 60 + "..."


def test_create_conversation_returns_an_id_and_starts_untitled():
    graph = FakeGraph()
    conversation_id = create_conversation(graph, kind="code")

    assert conversation_id
    _, params = graph.calls[0]
    assert params["id"] == conversation_id
    assert params["kind"] == "code"
    assert params["title"] == UNTITLED
    assert params["timestamp"]


def test_append_message_carries_role_and_mode():
    graph = FakeGraph()
    append_message(graph, "conv-1", "assistant", "the answer", mode="Graph RAG")

    _, params = graph.calls[0]
    assert params["conversation_id"] == "conv-1"
    assert params["role"] == "assistant"
    assert params["content"] == "the answer"
    assert params["mode"] == "Graph RAG"


def test_append_message_positions_by_existing_count():
    """Ordering must not depend on timestamps - see load_messages."""
    graph = FakeGraph()
    append_message(graph, "conv-1", "user", "question")

    cypher, _ = graph.calls[0]
    assert "count(existing) AS position" in cypher
    assert "position: position" in cypher


def test_load_messages_orders_by_position():
    graph = FakeGraph()
    load_messages(graph, "conv-1")

    cypher, _ = graph.calls[0]
    assert "ORDER BY m.position ASC" in cypher


def test_list_conversations_maps_rows():
    graph = FakeGraph(
        rows=[
            {
                "id": "conv-1",
                "kind": "code",
                "title": "how does retrieval work",
                "subject": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
                "message_count": 4,
            }
        ]
    )
    conversations = list_conversations(graph)

    assert conversations == [
        Conversation(
            id="conv-1",
            kind="code",
            title="how does retrieval work",
            subject=None,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-02T00:00:00+00:00",
            message_count=4,
        )
    ]


def test_rename_conversation_falls_back_on_a_blank_title():
    graph = FakeGraph()
    rename_conversation(graph, "conv-1", "   ")

    _, params = graph.calls[0]
    assert params["title"] == UNTITLED


def _message(role, content):
    return Message(id=f"{role}-{content}", role=role, content=content, created_at="t")


def test_chat_history_pairs_questions_with_answers():
    messages = [
        _message("user", "q1"),
        _message("assistant", "a1"),
        _message("user", "q2"),
        _message("assistant", "a2"),
    ]
    assert chat_history(messages) == [("q1", "a1"), ("q2", "a2")]


def test_chat_history_drops_a_question_with_no_answer():
    """A failed turn saves the question but never an answer."""
    messages = [
        _message("user", "q1"),
        _message("assistant", "a1"),
        _message("user", "unanswered"),
    ]
    assert chat_history(messages) == [("q1", "a1")]


def test_chat_history_keeps_only_the_most_recent_turns():
    messages = []
    for i in range(10):
        messages.append(_message("user", f"q{i}"))
        messages.append(_message("assistant", f"a{i}"))

    assert chat_history(messages, limit=2) == [("q8", "a8"), ("q9", "a9")]


def test_chat_history_of_an_empty_conversation():
    assert chat_history([]) == []
