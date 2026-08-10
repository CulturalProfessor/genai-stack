import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";
import type { Message } from "@/lib/api";

type Props = {
  messages: Message[];
  streaming: string | null;
  error: string | null;
};

export function Transcript({ messages, streaming, error }: Props) {
  const end = useRef<HTMLDivElement>(null);

  // Follow the answer as it arrives, the way a chat is expected to behave.
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, streaming]);

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-8">
      {messages.map((message) =>
        message.role === "user" ? (
          <div key={message.id} className="mb-6 flex justify-end">
            <div className="max-w-[85%] animate-fade-in rounded-2xl rounded-br-sm border border-white/[0.06] bg-secondary px-4 py-2.5 text-[0.95rem] leading-relaxed">
              {message.content}
            </div>
          </div>
        ) : (
          <div key={message.id} className="mb-8 animate-fade-in">
            {message.mode && (
              <p className="mb-1.5 text-[0.7rem] uppercase tracking-wider text-muted-foreground/70">
                {message.mode}
              </p>
            )}
            <Answer content={message.content} />
          </div>
        ),
      )}

      {streaming !== null && (
        <div className="mb-8">
          {streaming === "" ? (
            <Thinking />
          ) : (
            <>
              <Answer content={streaming} />
              <span className="ml-0.5 inline-block h-4 w-[2px] animate-blink bg-foreground align-middle" />
            </>
          )}
        </div>
      )}

      {error && (
        <div className="mb-8 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm">
          {error}
        </div>
      )}

      <div ref={end} />
    </div>
  );
}

function Answer({ content }: { content: string }) {
  return (
    <div className={cn("prose-answer")}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  );
}

function Thinking() {
  return (
    <div className="flex items-center gap-1.5 text-muted-foreground">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="size-1.5 animate-bounce rounded-full bg-current"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}
