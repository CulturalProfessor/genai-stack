import { useEffect, useRef, useState } from "react";
import { FileText, Trash2, Upload } from "lucide-react";

import { Composer } from "@/components/Composer";
import { Sidebar } from "@/components/Sidebar";
import { Transcript } from "@/components/Transcript";
import { Button } from "@/components/ui/button";
import { useChat } from "@/hooks/useChat";
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
  type PdfDocument,
} from "@/lib/api";
import { cn } from "@/lib/utils";

export function PdfView() {
  const [documents, setDocuments] = useState<PdfDocument[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const picker = useRef<HTMLInputElement>(null);

  // null while the documents are still loading, so the chat hook waits rather
  // than creating a conversation for a document that is not chosen yet.
  const chat = useChat("pdf", loaded ? selected : null);

  useEffect(() => {
    listDocuments()
      .then((list) => {
        setDocuments(list);
        setSelected((current) => current ?? (list.length ? list[0].id : null));
        setLoaded(true);
      })
      .catch((exc: Error) => {
        setUploadError(exc.message);
        setLoaded(true);
      });
  }, []);

  const upload = async (file: File) => {
    setBusy(`Indexing ${file.name}...`);
    setUploadError(null);
    try {
      const document = await uploadDocument(file);
      const list = await listDocuments();
      setDocuments(list);
      setSelected(document.id);
    } catch (exc) {
      setUploadError((exc as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const remove = async (id: string) => {
    await deleteDocument(id);
    const list = await listDocuments();
    setDocuments(list);
    if (selected === id) setSelected(list.length ? list[0].id : null);
  };

  const document_ = documents.find((d) => d.id === selected) ?? null;
  const empty = chat.messages.length === 0 && chat.streaming === null;

  return (
    <>
      <Sidebar
        title="Chat with a PDF"
        conversations={chat.conversations}
        activeId={chat.activeId}
        onSelect={chat.setActiveId}
        onNew={chat.newChat}
        onRename={chat.rename}
        onDelete={chat.remove}
      >
        <div className="px-3 pb-2">
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
            variant="outline"
            className="w-full gap-2"
            disabled={busy !== null}
            onClick={() => picker.current?.click()}
          >
            <Upload /> {busy ? "Indexing..." : "Upload a PDF"}
          </Button>
          {uploadError && (
            <p className="mt-2 text-xs text-destructive">{uploadError}</p>
          )}

          <p className="px-1 pb-1 pt-4 text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/70">
            Documents
          </p>
          {documents.length === 0 && (
            <p className="px-1 text-xs text-muted-foreground">Nothing uploaded yet.</p>
          )}
          <div className="max-h-48 overflow-y-auto">
            {documents.map((item) => (
              <div
                key={item.id}
                className={cn(
                  "group flex items-center gap-1 rounded-md px-1",
                  selected === item.id ? "bg-sidebar-accent" : "hover:bg-white/[0.05]",
                )}
              >
                <button
                  onClick={() => setSelected(item.id)}
                  title={`${item.name} — ${item.pages} pages, ${item.chunks} chunks`}
                  className="flex min-w-0 flex-1 items-center gap-2 py-1.5 pl-1 text-left text-sm"
                >
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate">{item.name}</span>
                </button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Delete ${item.name}`}
                  className="size-7 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={() => remove(item.id)}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            ))}
          </div>
        </div>
      </Sidebar>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto">
          {!document_ ? (
            <div className="mx-auto flex h-full max-w-md flex-col justify-center px-4 text-center">
              <h1 className="mb-2 text-2xl font-semibold tracking-tight">
                Chat with a PDF
              </h1>
              <p className="text-sm text-muted-foreground">
                Upload one from the sidebar. Documents are kept until you delete
                them, each with its own chats.
              </p>
            </div>
          ) : empty ? (
            <div className="mx-auto flex h-full max-w-md flex-col justify-center px-4 text-center">
              <h1 className="mb-2 text-xl font-semibold tracking-tight">
                {document_.name}
              </h1>
              <p className="text-sm text-muted-foreground">
                {document_.pages} pages, {document_.chunks} chunks. Answers come
                from this document only.
              </p>
              {chat.error && (
                <p className="mt-6 text-sm text-destructive">{chat.error}</p>
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

        {document_ && (
          <Composer
            onSend={(text) => chat.send(text)}
            onStop={chat.stopStreaming}
            busy={chat.busy}
            placeholder={`Ask about ${document_.name}`}
            hint="Answers come from the selected document only."
          />
        )}
      </main>
    </>
  );
}
