import { useCallback, useEffect, useRef, useState } from "react";

import { Composer } from "@/components/Composer";
import { Sidebar } from "@/components/Sidebar";
import { Transcript } from "@/components/Transcript";
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

const STARTERS = [
  "What does this codebase do?",
  "How does error handling work?",
  "Where do requests get retried?",
  "Which files depend on each other most?",
];

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"Graph RAG" | "LLM only">("Graph RAG");
  const stop = useRef<(() => void) | null>(null);

  const refresh = useCallback(async () => {
    const list = await listConversations();
    setConversations(list);
    return list;
  }, []);

  // On load, open the most recent chat, or start one if there is none.
  useEffect(() => {
    (async () => {
      try {
        const list = await refresh();
        if (list.length) {
          setActiveId(list[0].id);
        } else {
          const created = await createConversation();
          setConversations([created]);
          setActiveId(created.id);
        }
      } catch (exc) {
        setError(`Could not reach the API: ${(exc as Error).message}`);
      }
    })();
  }, [refresh]);

  useEffect(() => {
    if (!activeId) return;
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
    const created = await createConversation();
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
  };

  const send = (text: string) => {
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
    stop.current = ask(activeId, text, mode === "Graph RAG", {
      onToken: (token) => {
        answer += token;
        setStreaming(answer);
      },
      onDone: async () => {
        setStreaming(null);
        stop.current = null;
        // Reload from the store rather than trusting the local copy: the
        // server is what assigned ids, positions and the title.
        const [{ messages }] = await Promise.all([
          getConversation(activeId),
          refresh(),
        ]);
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
    refresh();
  };

  const remove = async (id: string) => {
    await deleteConversation(id);
    const list = await refresh();
    if (id === activeId) setActiveId(list.length ? list[0].id : null);
  };

  const empty = messages.length === 0 && streaming === null;

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={newChat}
        onRename={rename}
        onDelete={remove}
        status={null}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto">
          {empty ? (
            <div className="mx-auto flex h-full w-full max-w-3xl flex-col justify-center px-4">
              <h1 className="mb-2 text-center text-2xl font-semibold tracking-tight">
                Ask the codebase
              </h1>
              <p className="mb-8 text-center text-sm text-muted-foreground">
                Answers come from the indexed code and the call graph around it.
              </p>
              <div className="grid gap-2 sm:grid-cols-2">
                {STARTERS.map((starter) => (
                  <button
                    key={starter}
                    onClick={() => send(starter)}
                    className="rounded-xl border border-border px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:bg-white/[0.03] hover:text-foreground"
                  >
                    {starter}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <Transcript messages={messages} streaming={streaming} error={error} />
          )}
        </div>

        <Composer
          onSend={send}
          onStop={stopStreaming}
          busy={streaming !== null}
          mode={mode}
          onModeChange={setMode}
        />
      </main>
    </div>
  );
}
