import { useEffect, useRef, useState } from "react";
import { Loader2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  clearGraph,
  getGraph,
  importIndexFile,
  indexRepository,
  type GraphSummary,
  type IndexProgress,
} from "@/lib/api";

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-2xl font-semibold tabular-nums tracking-tight">{value}</p>
      {note && <p className="text-[0.7rem] text-muted-foreground">{note}</p>}
    </div>
  );
}

function describe(progress: IndexProgress): string {
  switch (progress.stage) {
    case "cloning":
      return `Cloning ${progress.repo}...`;
    case "parsing":
      return `Parsing ${progress.repo} with tree-sitter...`;
    case "parsed":
      return `Found ${progress.symbols.toLocaleString()} symbols across ${progress.files.toLocaleString()} files`;
    case "importing":
      return `Embedded and imported ${progress.imported.toLocaleString()} / ${progress.total.toLocaleString()} symbols`;
  }
}

export function IndexView() {
  const [graph, setGraph] = useState<GraphSummary | null>(null);
  const [url, setUrl] = useState("");
  const [branch, setBranch] = useState("");
  const [progress, setProgress] = useState<IndexProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const stop = useRef<(() => void) | null>(null);
  const picker = useRef<HTMLInputElement>(null);

  const load = () => getGraph().then(setGraph).catch((exc: Error) => setError(exc.message));

  useEffect(() => {
    load();
    return () => stop.current?.();
  }, []);

  const start = () => {
    if (!url.trim()) return;
    setError(null);
    setDone(null);
    setProgress({ stage: "cloning", repo: url.trim() });
    stop.current = indexRepository(url.trim(), branch.trim(), {
      onProgress: setProgress,
      onDone: (imported, repo) => {
        setProgress(null);
        setDone(`Imported ${imported.toLocaleString()} symbols from ${repo}`);
        setUrl("");
        load();
      },
      onError: (message) => {
        setProgress(null);
        setError(message);
      },
    });
  };

  const importFile = async (file: File) => {
    setError(null);
    setDone(null);
    setImporting(true);
    try {
      const { imported, repo } = await importIndexFile(file);
      setDone(`Imported ${imported.toLocaleString()} symbols from ${repo}`);
      load();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setImporting(false);
    }
  };

  const percent =
    progress?.stage === "importing" && progress.total
      ? Math.round((progress.imported / progress.total) * 100)
      : null;

  return (
    <main className="flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl px-6 py-10">
        <h1 className="text-2xl font-semibold tracking-tight">Index a repository</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Parsed with tree-sitter, never executed — no install or build step runs.
        </p>

        {graph && graph.counts.symbols > 0 ? (
          <>
            <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Files" value={graph.counts.files.toLocaleString()} />
              <Stat label="Symbols" value={graph.counts.symbols.toLocaleString()} />
              <Stat
                label="Call edges"
                value={graph.counts.calls.toLocaleString()}
                note={`${graph.counts.cross_file.toLocaleString()} cross-file`}
              />
              <Stat label="Import edges" value={graph.counts.imports.toLocaleString()} />
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              Cross-file call edges are what retrieval traverses to reach code a
              question does not name directly.
            </p>

            <div className="mt-6 overflow-hidden rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead className="bg-white/[0.03] text-left text-xs text-muted-foreground">
                  <tr>
                    <th className="px-4 py-2 font-medium">Repository</th>
                    <th className="px-4 py-2 font-medium">Files</th>
                    <th className="px-4 py-2 font-medium">Symbols</th>
                    <th className="px-4 py-2 font-medium">Indexed</th>
                  </tr>
                </thead>
                <tbody>
                  {graph.repositories.map((repo) => (
                    <tr key={repo.name} className="border-t border-border">
                      <td className="px-4 py-2">
                        {repo.source_url ? (
                          <a
                            href={repo.source_url.replace(/\.git$/, "")}
                            target="_blank"
                            rel="noreferrer"
                            className="text-primary hover:underline"
                          >
                            {repo.name}
                          </a>
                        ) : (
                          repo.name
                        )}
                      </td>
                      <td className="px-4 py-2 tabular-nums">{repo.files}</td>
                      <td className="px-4 py-2 tabular-nums">{repo.symbols}</td>
                      <td className="px-4 py-2 text-muted-foreground">
                        {repo.indexed_at.slice(0, 16).replace("T", " ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <div className="mt-8 rounded-xl border border-border bg-card px-4 py-6 text-center text-sm text-muted-foreground">
            Nothing is indexed yet. Add a repository below, then ask questions
            about it.
          </div>
        )}

        <div className="mt-10 rounded-xl border border-border bg-card p-5">
          <h2 className="text-sm font-semibold">Add a repository</h2>
          <div className="mt-3 flex flex-col gap-2 sm:flex-row">
            <Input
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && start()}
              placeholder="https://github.com/owner/repo"
              className="flex-[3]"
              disabled={progress !== null}
            />
            <Input
              value={branch}
              onChange={(event) => setBranch(event.target.value)}
              placeholder="default branch"
              className="flex-1"
              disabled={progress !== null}
            />
            <Button onClick={start} disabled={progress !== null || !url.trim()}>
              {progress ? <Loader2 className="animate-spin" /> : null}
              {progress ? "Indexing" : "Index"}
            </Button>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Public repositories only, from {graph?.allowed_hosts.join(", ") ?? "..."}.
            Only .ts, .tsx, .js and .jsx files are understood.
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-border pt-4">
            <input
              ref={picker}
              type="file"
              accept="application/json,.json"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) importFile(file);
                event.target.value = "";
              }}
            />
            <Button
              variant="outline"
              disabled={progress !== null || importing}
              onClick={() => picker.current?.click()}
            >
              <Upload /> {importing ? "Importing..." : "Import an index file"}
            </Button>
            <span className="text-xs text-muted-foreground">
              For a local checkout: <code className="text-[0.7rem]">python3 -m
              genai_stack.extract --repo /path/to/repo</code>, then import the JSON.
            </span>
          </div>

          {progress && (
            <div className="mt-4">
              <p className="text-sm">{describe(progress)}</p>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/[0.07]">
                <div
                  className="h-full rounded-full bg-primary transition-all duration-300"
                  style={{ width: percent === null ? "25%" : `${percent}%` }}
                />
              </div>
            </div>
          )}
          {done && <p className="mt-4 text-sm text-primary">{done}</p>}
          {error && <p className="mt-4 text-sm text-destructive">{error}</p>}
        </div>

        <details className="mt-8 rounded-xl border border-border px-4 py-3">
          <summary className="cursor-pointer text-sm text-muted-foreground">
            Danger zone
          </summary>
          <p className="mt-3 text-xs text-muted-foreground">
            Removes every File, Code and Repository node so you can re-import
            cleanly. Conversations and uploaded PDFs are not touched.
          </p>
          <Button
            variant="destructive"
            className="mt-3"
            onClick={async () => {
              await clearGraph();
              load();
            }}
          >
            Delete imported graph
          </Button>
        </details>
      </div>
    </main>
  );
}
