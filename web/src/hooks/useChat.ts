import { useCallback, useEffect, useRef, useState } from "react";

import {
  ask,
  attachDocument,
  createConversation,
  deleteConversation,
  detachDocument,
  getConversation,
  listConversations,
  renameConversation,
  type Conversation,
  type Message,
  type PdfDocument,
} from "@/lib/api";

/** Everything the chat needs: its conversations, its messages, its PDFs. */
export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  // The PDFs uploaded into the open chat. Questions are answered from these
  // alongside the code, so they belong to the conversation, not to the view.
  const [documents, setDocuments] = useState<PdfDocument[]>([]);
  // Partial answers keyed by the conversation they belong to. An answer keeps
  // arriving after you start a new chat, so "is something streaming" is a
  // question about a conversation, not about the view.
  const [streams, setStreams] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const closers = useRef<Record<string, () => void>>({});
  // Read inside stream callbacks, which close over the activeId of the render
  // that started them - by the time an answer lands it may be stale.
  const openId = useRef<string | null>(null);

  const refresh = useCallback(async () => {
    const list = await listConversations();
    setConversations(list);
    return list;
  }, []);

  // Open the most recent conversation, or start one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await listConversations();
        if (cancelled) return;
        setConversations(list);
        if (list.length) {
          setActiveId(list[0].id);
        } else {
          const created = await createConversation();
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
  }, []);

  useEffect(() => {
    openId.current = activeId;
  }, [activeId]);

  // Nothing is listening once this view goes away, and an EventSource left
  // open would keep the connection for the life of the page.
  useEffect(
    () => () => {
      Object.values(closers.current).forEach((close) => close());
      closers.current = {};
    },
    [],
  );

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setDocuments([]);
      return;
    }
    let cancelled = false;
    getConversation(activeId)
      .then(({ messages, documents }) => {
        if (!cancelled) {
          setMessages(messages);
          setDocuments(documents);
          setError(null);
        }
      })
      .catch((exc: Error) => !cancelled && setError(exc.message));
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  const newChat = async () => {
    // Reuse an empty chat rather than piling up identical blank ones. One with
    // a PDF already uploaded into it is not empty, whatever its messages say.
    const empty = conversations.find(
      (c) => c.message_count === 0 && c.document_count === 0,
    );
    if (empty) {
      setActiveId(empty.id);
      return;
    }
    const created = await createConversation();
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
  };

  // Both refresh the list: a chat's document count is what stops "New chat"
  // from handing back a blank one that already has a PDF in it.
  const attach = async (file: File) => {
    if (!activeId) return;
    const attached = await attachDocument(activeId, file);
    setDocuments((current) =>
      current.some((d) => d.id === attached.id) ? current : [...current, attached],
    );
    await refresh();
  };

  const detach = async (id: string) => {
    if (!activeId) return;
    await detachDocument(activeId, id);
    setDocuments((current) => current.filter((d) => d.id !== id));
    await refresh();
  };

  const send = (text: string, rag = true) => {
    const target = activeId;
    if (!target || closers.current[target]) return;
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
    setStreams((current) => ({ ...current, [target]: "" }));

    const finish = (next: Record<string, string>, id: string) => {
      const { [id]: _gone, ...rest } = next;
      return rest;
    };

    let answer = "";
    closers.current[target] = ask(target, text, rag, {
      onToken: (token) => {
        answer += token;
        setStreams((current) =>
          // Dropped by a delete or a stop mid-flight: do not resurrect it.
          target in current ? { ...current, [target]: answer } : current,
        );
      },
      onDone: async () => {
        setStreams((current) => finish(current, target));
        delete closers.current[target];
        // Reload rather than trusting the local copy: the server assigned the
        // ids, the positions and the title. Only paint it if that conversation
        // is still the one on screen - otherwise this would drop one chat's
        // messages into another.
        const [{ messages: saved }] = await Promise.all([
          getConversation(target),
          refresh(),
        ]);
        if (openId.current === target) setMessages(saved);
      },
      onError: (message) => {
        setStreams((current) => finish(current, target));
        delete closers.current[target];
        if (openId.current === target) setError(message);
      },
    });
  };

  const stopStreaming = () => {
    if (!activeId) return;
    closers.current[activeId]?.();
    delete closers.current[activeId];
    setStreams((current) => {
      const { [activeId]: _gone, ...rest } = current;
      return rest;
    });
  };

  const rename = async (id: string, title: string) => {
    await renameConversation(id, title);
    await refresh();
  };

  const remove = async (id: string) => {
    closers.current[id]?.();
    delete closers.current[id];
    setStreams((current) => {
      const { [id]: _gone, ...rest } = current;
      return rest;
    });
    await deleteConversation(id);
    const list = await refresh();
    if (id === activeId) setActiveId(list.length ? list[0].id : null);
  };

  return {
    conversations,
    activeId,
    setActiveId,
    messages,
    // Only the open conversation's answer is on screen; one still arriving for
    // another chat keeps arriving, quietly.
    streaming: activeId && activeId in streams ? streams[activeId] : null,
    busy: !!activeId && activeId in streams,
    error,
    setError,
    documents,
    attach,
    detach,
    newChat,
    send,
    stopStreaming,
    rename,
    remove,
  };
}
