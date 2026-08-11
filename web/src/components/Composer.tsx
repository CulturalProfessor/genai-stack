import { useEffect, useRef, useState } from "react";
import { ArrowUp, FileText, Paperclip, Square, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { PdfDocument } from "@/lib/api";

type Props = {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  placeholder?: string;
  mode: "Graph RAG" | "LLM only";
  onModeChange: (mode: "Graph RAG" | "LLM only") => void;
  /** PDFs already in this chat, and the two ways to change that. */
  documents: PdfDocument[];
  onAttach: (file: File) => Promise<void>;
  onDetach: (id: string) => void;
};

const MODES = ["Graph RAG", "LLM only"] as const;

export function Composer({
  onSend,
  onStop,
  busy,
  mode,
  onModeChange,
  documents,
  onAttach,
  onDetach,
  placeholder = "Ask about the indexed code",
}: Props) {
  const [text, setText] = useState("");
  const [uploading, setUploading] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const picker = useRef<HTMLInputElement>(null);

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

  // Indexing happens during the upload, so a large PDF holds this for a while.
  const upload = async (file: File) => {
    setUploading(file.name);
    setUploadError(null);
    try {
      await onAttach(file);
    } catch (exc) {
      setUploadError((exc as Error).message);
    } finally {
      setUploading(null);
    }
  };

  return (
    <div className="sticky bottom-0 bg-gradient-to-t from-background via-background to-transparent pb-4 pt-6">
      <div className="mx-auto w-full max-w-3xl px-4">
        <div className="rounded-2xl border border-border bg-card shadow-lg shadow-black/30 focus-within:border-primary/50">
          {(documents.length > 0 || uploading) && (
            <div className="flex flex-wrap gap-1.5 px-3 pt-3">
              {documents.map((pdf) => (
                <span
                  key={pdf.id}
                  title={`${pdf.pages} pages, ${pdf.chunks} chunks`}
                  className="flex items-center gap-1.5 rounded-lg bg-white/[0.06] py-1 pl-2 pr-1 text-xs"
                >
                  <FileText className="size-3 shrink-0 text-muted-foreground" />
                  <span className="max-w-[14rem] truncate">{pdf.name}</span>
                  <button
                    onClick={() => onDetach(pdf.id)}
                    aria-label={`Remove ${pdf.name}`}
                    className="rounded p-0.5 text-muted-foreground hover:bg-white/[0.08] hover:text-foreground"
                  >
                    <X className="size-3" />
                  </button>
                </span>
              ))}
              {uploading && (
                <span className="flex items-center gap-1.5 rounded-lg bg-white/[0.06] px-2 py-1 text-xs text-muted-foreground">
                  <FileText className="size-3 shrink-0" />
                  <span className="max-w-[14rem] truncate">{uploading}</span>
                  <span>- indexing...</span>
                </span>
              )}
            </div>
          )}

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
            <div className="flex items-center gap-1.5">
              <input
                ref={picker}
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) upload(file);
                  event.target.value = "";
                }}
              />
              <Button
                variant="ghost"
                size="icon"
                aria-label="Attach a PDF"
                title="Attach a PDF"
                disabled={uploading !== null}
                className="size-8 text-muted-foreground hover:text-foreground"
                onClick={() => picker.current?.click()}
              >
                <Paperclip className="size-4" />
              </Button>

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
            </div>

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

        {uploadError ? (
          <p className="mt-2 text-center text-[0.7rem] text-destructive">
            {uploadError}
          </p>
        ) : (
          <p className="mt-2 text-center text-[0.7rem] text-muted-foreground/70">
            {mode === "Graph RAG"
              ? "Graph RAG searches the indexed code and follows its call and import edges."
              : "Answers come from the model alone."}
            {documents.length > 0 &&
              ` Attached PDFs are searched too, and stay with this chat.`}
          </p>
        )}
      </div>
    </div>
  );
}
