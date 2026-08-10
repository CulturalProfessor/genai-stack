import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Props = {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  placeholder?: string;
  hint?: string;
  /** Omitted by the PDF chat, which has only one way to answer. */
  mode?: "Graph RAG" | "LLM only";
  onModeChange?: (mode: "Graph RAG" | "LLM only") => void;
};

const MODES = ["Graph RAG", "LLM only"] as const;

export function Composer({
  onSend,
  onStop,
  busy,
  mode,
  onModeChange,
  placeholder = "Ask about the indexed code",
  hint = "Graph RAG searches the indexed code and follows its call and import edges.",
}: Props) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);

  // Grow with the question instead of scrolling a two-line box.
  useEffect(() => {
    const element = box.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [text]);

  const send = () => {
    const question = text.trim();
    if (!question || busy) return;
    onSend(question);
    setText("");
  };

  return (
    <div className="sticky bottom-0 bg-gradient-to-t from-background via-background to-transparent pb-4 pt-6">
      <div className="mx-auto w-full max-w-3xl px-4">
        <div className="rounded-2xl border border-border bg-card shadow-lg shadow-black/30 focus-within:border-primary/50">
          <textarea
            ref={box}
            rows={1}
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            placeholder={placeholder}
            className="max-h-[200px] w-full resize-none bg-transparent px-4 pt-3.5 text-[0.95rem] outline-none placeholder:text-muted-foreground"
          />

          <div className="flex items-center justify-between gap-2 px-3 pb-2.5 pt-1">
            {mode && onModeChange ? (
              <div className="flex rounded-lg bg-white/[0.04] p-0.5">
                {MODES.map((option) => (
                  <button
                    key={option}
                    onClick={() => onModeChange(option)}
                    className={cn(
                      "rounded-md px-2.5 py-1 text-xs transition-colors",
                      mode === option
                        ? "bg-primary/20 text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {option}
                  </button>
                ))}
              </div>
            ) : (
              <span />
            )}

            {busy ? (
              <Button size="icon" variant="secondary" onClick={onStop} aria-label="Stop">
                <Square className="size-3 fill-current" />
              </Button>
            ) : (
              <Button
                size="icon"
                onClick={send}
                disabled={!text.trim()}
                aria-label="Send"
              >
                <ArrowUp />
              </Button>
            )}
          </div>
        </div>

        <p className="mt-2 text-center text-[0.7rem] text-muted-foreground/70">
          {hint}
        </p>
      </div>
    </div>
  );
}
