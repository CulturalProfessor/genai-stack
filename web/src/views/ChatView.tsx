import { useState } from "react";

import { Composer } from "@/components/Composer";
import { Sidebar } from "@/components/Sidebar";
import { Transcript } from "@/components/Transcript";
import { useChat } from "@/hooks/useChat";

const STARTERS = [
  "What does this codebase do?",
  "How does error handling work?",
  "Where do requests get retried?",
  "Which files depend on each other most?",
];

export function ChatView({ model }: { model: string | null }) {
  const chat = useChat("code");
  const [mode, setMode] = useState<"Graph RAG" | "LLM only">("Graph RAG");

  const empty = chat.messages.length === 0 && chat.streaming === null;

  return (
    <>
      <Sidebar
        title="Ask the codebase"
        conversations={chat.conversations}
        activeId={chat.activeId}
        onSelect={chat.setActiveId}
        onNew={chat.newChat}
        onRename={chat.rename}
        onDelete={chat.remove}
        footer={model ? <p className="truncate">Model: {model}</p> : null}
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
                    onClick={() => chat.send(starter, mode === "Graph RAG")}
                    className="rounded-xl border border-border px-4 py-3 text-left text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:bg-white/[0.03] hover:text-foreground"
                  >
                    {starter}
                  </button>
                ))}
              </div>
              {chat.error && (
                <p className="mt-6 text-center text-sm text-destructive">{chat.error}</p>
              )}
            </div>
          ) : (
            <Transcript
              messages={chat.messages}
              streaming={chat.streaming}
              error={chat.error}
            />
          )}
        </div>

        <Composer
          onSend={(text) => chat.send(text, mode === "Graph RAG")}
          onStop={chat.stopStreaming}
          busy={chat.streaming !== null}
          mode={mode}
          onModeChange={setMode}
        />
      </main>
    </>
  );
}
