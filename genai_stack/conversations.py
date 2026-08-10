"""Saved conversations, stored in Neo4j alongside the code graph.

Neo4j is already a hard dependency of every service here, so chats live there
rather than in a second store: no extra volume to mount, and a conversation
survives `docker compose down`. The labels are deliberately outside the code
graph's own - `reset_graph` deletes File and Code nodes, and clearing an index
must not take a user's chat history with it.

    (:Conversation {id, kind, title, created_at, updated_at, subject})
    (:Message {id, role, content, mode, created_at, position})

    (:Conversation)-[:HAS_MESSAGE]->(:Message)

`kind` separates the code chat from the PDF chat, which are different pages
over the same store. `subject` is the PDF's document id for a pdf conversation
and unset for a code one.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

CONVERSATION_LABEL = "Conversation"
MESSAGE_LABEL = "Message"

KIND_CODE = "code"
KIND_PDF = "pdf"

# A new chat has no title until its first question arrives.
UNTITLED = "New chat"

# Titles are the question's first line, cut to something a sidebar can show.
TITLE_MAX_CHARS = 60


@dataclass(frozen=True)
class Conversation:
    id: str
    kind: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    subject: str | None = None


@dataclass(frozen=True)
class Message:
    id: str
    role: str
    content: str
    created_at: str
    mode: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def title_from(question: str) -> str:
    """A sidebar label derived from the first question of a chat."""
    line = " ".join(question.strip().split())
    if not line:
        return UNTITLED
    if len(line) <= TITLE_MAX_CHARS:
        return line
    # Prefer breaking on a word boundary, but only if one is reasonably close
    # to the limit - otherwise a long unbroken token would cut to nearly nothing.
    cut = line[:TITLE_MAX_CHARS]
    space = cut.rfind(" ")
    if space > TITLE_MAX_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip() + "..."


def create_constraints(graph) -> None:
    """Called from apply_schema, so every entry point gets these."""
    graph.query(
        f"CREATE CONSTRAINT conversation_id IF NOT EXISTS "
        f"FOR (c:{CONVERSATION_LABEL}) REQUIRE c.id IS UNIQUE"
    )
    graph.query(
        f"CREATE CONSTRAINT message_id IF NOT EXISTS "
        f"FOR (m:{MESSAGE_LABEL}) REQUIRE m.id IS UNIQUE"
    )


def create_conversation(graph, kind: str = KIND_CODE, subject: str | None = None) -> str:
    """Start an empty conversation and return its id."""
    conversation_id = str(uuid.uuid4())
    timestamp = _now()
    graph.query(
        f"""
        CREATE (c:{CONVERSATION_LABEL} {{
            id: $id, kind: $kind, title: $title, subject: $subject,
            created_at: $timestamp, updated_at: $timestamp
        }})
        """,
        {
            "id": conversation_id,
            "kind": kind,
            "title": UNTITLED,
            "subject": subject,
            "timestamp": timestamp,
        },
    )
    return conversation_id


def list_conversations(
    graph, kind: str = KIND_CODE, subject: str | None = None
) -> list[Conversation]:
    """Most recently used first - the order a sidebar wants."""
    rows = graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{kind: $kind}})
        WHERE $subject IS NULL OR c.subject = $subject
        OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL})
        WITH c, count(m) AS message_count
        RETURN c.id AS id, c.kind AS kind, c.title AS title,
               c.subject AS subject, c.created_at AS created_at,
               c.updated_at AS updated_at, message_count
        ORDER BY c.updated_at DESC
        """,
        {"kind": kind, "subject": subject},
    )
    return [Conversation(**row) for row in rows]


def get_conversation(graph, conversation_id: str) -> Conversation | None:
    rows = graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{id: $id}})
        OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL})
        WITH c, count(m) AS message_count
        RETURN c.id AS id, c.kind AS kind, c.title AS title,
               c.subject AS subject, c.created_at AS created_at,
               c.updated_at AS updated_at, message_count
        """,
        {"id": conversation_id},
    )
    return Conversation(**rows[0]) if rows else None


def load_messages(graph, conversation_id: str) -> list[Message]:
    """Every message in the conversation, oldest first.

    Ordered by an explicit position rather than by timestamp: two messages of
    the same turn are written milliseconds apart, and a clock with coarse
    resolution would be free to swap the question and its answer.
    """
    rows = graph.query(
        f"""
        MATCH (:{CONVERSATION_LABEL} {{id: $id}})-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL})
        RETURN m.id AS id, m.role AS role, m.content AS content,
               m.mode AS mode, m.created_at AS created_at
        ORDER BY m.position ASC
        """,
        {"id": conversation_id},
    )
    return [Message(**row) for row in rows]


def append_message(
    graph,
    conversation_id: str,
    role: str,
    content: str,
    mode: str | None = None,
) -> str:
    """Add one message and bump the conversation's updated_at."""
    message_id = str(uuid.uuid4())
    timestamp = _now()
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{id: $conversation_id}})
        OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(existing:{MESSAGE_LABEL})
        WITH c, count(existing) AS position
        CREATE (c)-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL} {{
            id: $id, role: $role, content: $content, mode: $mode,
            position: position, created_at: $timestamp
        }})
        SET c.updated_at = $timestamp
        """,
        {
            "conversation_id": conversation_id,
            "id": message_id,
            "role": role,
            "content": content,
            "mode": mode,
            "timestamp": timestamp,
        },
    )
    return message_id


def rename_conversation(graph, conversation_id: str, title: str) -> None:
    graph.query(
        f"MATCH (c:{CONVERSATION_LABEL} {{id: $id}}) SET c.title = $title",
        {"id": conversation_id, "title": title.strip() or UNTITLED},
    )


def set_title_if_untitled(graph, conversation_id: str, question: str) -> None:
    """Name a chat after its first question, leaving a manual rename alone."""
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{id: $id}})
        WHERE c.title = $untitled
        SET c.title = $title
        """,
        {"id": conversation_id, "untitled": UNTITLED, "title": title_from(question)},
    )


def delete_conversation(graph, conversation_id: str) -> None:
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{id: $id}})
        OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL})
        DETACH DELETE c, m
        """,
        {"id": conversation_id},
    )


def delete_conversations_for(graph, subject: str) -> None:
    """Drop every conversation about a subject - used when a PDF is removed."""
    graph.query(
        f"""
        MATCH (c:{CONVERSATION_LABEL} {{subject: $subject}})
        OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:{MESSAGE_LABEL})
        DETACH DELETE c, m
        """,
        {"subject": subject},
    )


def chat_history(messages: list[Message], limit: int = 6) -> list[tuple[str, str]]:
    """Recent turns as (question, answer) pairs for a chain's chat_history.

    Capped because the whole history shares the model's context window with the
    retrieved code; the last few exchanges are what follow-up questions need.
    """
    pairs: list[tuple[str, str]] = []
    pending: str | None = None
    for message in messages:
        if message.role == "user":
            pending = message.content
        elif message.role == "assistant" and pending is not None:
            pairs.append((pending, message.content))
            pending = None
    return pairs[-limit:]
