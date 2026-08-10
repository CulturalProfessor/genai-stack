import { useCallback, useEffect, useRef, useState } from "react";

import {
  ask,
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
  type Conversation,
  type Message,
} from "@/lib/api";

/**
 * Everything a chat view needs, for one kind of conversation.
 *
 * The code chat and the PDF chat differ only in `kind` and whether they are
 * scoped to a subject, so they share this rather than the logic twice.
 * Passing subject === null means "not ready yet" (no PDF picked), which is
 * different from undefined ("not scoped to anything").
 */
export function useChat(kind: string, subject?: string | null) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const stop = useRef<(() => void) | null>(null);

  const scoped = subject ?? undefined;
  const ready = subject !== null;

  const refresh = useCallback(async () => {
    const list = await listConversations(kind, scoped);
    setConversations(list);
    return list;
  }, [kind, scoped]);

  // Open the most recent conversation for this scope, or start one.
  useEffect(() => {
    if (!ready) {
      setConversations([]);
      setActiveId(null);
      setMessages([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const list = await listConversations(kind, scoped);
        if (cancelled) return;
        setConversations(list);
        if (list.length) {
          setActiveId(list[0].id);
        } else {
          const created = await createConversation(kind, scoped);
          if (cancelled) return;
          setConversations([created]);
          setActiveId(created.id);
        }
        setError(null);
      } catch (exc) {
        if (!cancelled) setError(`Could not reach the API: ${(exc as Error).message}`);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [kind, scoped, ready]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    getConversation(activeId)
      .then(({ messages }) => {
        if (!cancelled) {
          setMessages(messages);
          setError(null);
        }
      })
      .catch((exc: Error) => !cancelled && setError(exc.message));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  const newChat = async () => {
    // Reuse an empty chat rather than piling up identical blank ones.
    const empty = conversations.find((c) => c.message_count === 0);
    if (empty) {
      setActiveId(empty.id);
      return;
    }
    const created = await createConversation(kind, scoped);
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
  };

  const send = (text: string, rag = true) => {
    if (!activeId) return;
    setError(null);
    setMessages((current) => [
      ...current,
      {
        id: `local-${Date.now()}`,
        role: "user",
        content: text,
        mode: null,
        created_at: new Date().toISOString(),
      },
    ]);
    setStreaming("");

    let answer = "";
    stop.current = ask(activeId, text, rag, {
      onToken: (token) => {
        answer += token;
        setStreaming(answer);
      },
      onDone: async () => {
        setStreaming(null);
        stop.current = null;
        // Reload rather than trusting the local copy: the server assigned the
        // ids, the positions and the title.
        const [{ messages }] = await Promise.all([getConversation(activeId), refresh()]);
        setMessages(messages);
      },
      onError: (message) => {
        setStreaming(null);
        stop.current = null;
        setError(message);
      },
    });
  };

  const stopStreaming = () => {
    stop.current?.();
    stop.current = null;
    setStreaming(null);
  };

  const rename = async (id: string, title: string) => {
    await renameConversation(id, title);
    await refresh();
  };

  const remove = async (id: string) => {
    await deleteConversation(id);
    const list = await refresh();
    if (id === activeId) setActiveId(list.length ? list[0].id : null);
  };

  return {
    conversations,
    activeId,
    setActiveId,
    messages,
    streaming,
    error,
    setError,
    newChat,
    send,
    stopStreaming,
    rename,
    remove,
  };
}
