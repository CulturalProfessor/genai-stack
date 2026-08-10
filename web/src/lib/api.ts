/**
 * The FastAPI service at :8504. Every call goes through here so the base URL
 * and the optional API key live in one place.
 */
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8504";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export type Conversation = {
  id: string;
  kind: string;
  title: string;
  subject: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode: string | null;
  created_at: string;
};

function headers(extra: Record<string, string> = {}) {
  return API_KEY ? { ...extra, "X-API-Key": API_KEY } : extra;
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function listConversations(
  kind = "code",
  subject?: string,
): Promise<Conversation[]> {
  const params = new URLSearchParams({ kind });
  if (subject) params.set("subject", subject);
  const response = await fetch(`${BASE}/conversations?${params}`, {
    headers: headers(),
  });
  const body = await json<{ conversations: Conversation[] }>(response);
  return body.conversations;
}

export async function createConversation(
  kind = "code",
  subject?: string,
): Promise<Conversation> {
  const response = await fetch(`${BASE}/conversations`, {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ kind, subject: subject ?? null }),
  });
  return json<Conversation>(response);
}

export async function getConversation(
  id: string,
): Promise<{ conversation: Conversation; messages: Message[] }> {
  const response = await fetch(`${BASE}/conversations/${id}`, { headers: headers() });
  return json(response);
}

export async function renameConversation(id: string, title: string): Promise<Conversation> {
  const response = await fetch(`${BASE}/conversations/${id}`, {
    method: "PATCH",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ title }),
  });
  return json<Conversation>(response);
}

export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${BASE}/conversations/${id}`, {
    method: "DELETE",
    headers: headers(),
  });
  await json(response);
}

export type AskHandlers = {
  onToken: (token: string) => void;
  onDone: (title: string) => void;
  onError: (message: string) => void;
};

/**
 * Ask a question and stream the answer.
 *
 * Server-sent events rather than fetch streaming because EventSource cannot
 * send headers, which is also why the key travels as a query parameter - the
 * API accepts it there for exactly this reason.
 */
export function ask(
  conversationId: string,
  text: string,
  rag: boolean,
  handlers: AskHandlers,
): () => void {
  const params = new URLSearchParams({ text, rag: String(rag) });
  if (API_KEY) params.set("api_key", API_KEY);
  const source = new EventSource(
    `${BASE}/conversations/${conversationId}/ask?${params.toString()}`,
  );

  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.token) handlers.onToken(payload.token);
    if (payload.error) {
      handlers.onError(payload.error);
      source.close();
    }
    if (payload.done) {
      handlers.onDone(payload.title ?? "");
      source.close();
    }
  };
  source.onerror = () => {
    // EventSource retries by itself; for a one-shot answer a drop is the end.
    handlers.onError("The connection to the API was lost.");
    source.close();
  };

  return () => source.close();
}

// ---------------------------------------------------------------------------
// The code graph
// ---------------------------------------------------------------------------

export type GraphCounts = {
  files: number;
  symbols: number;
  calls: number;
  cross_file: number;
  imports: number;
};

export type Repository = {
  name: string;
  source_url: string | null;
  files: number;
  symbols: number;
  indexed_at: string;
};

export type GraphSummary = {
  counts: GraphCounts;
  repositories: Repository[];
  allowed_hosts: string[];
  model: string;
};

export async function getGraph(): Promise<GraphSummary> {
  const response = await fetch(`${BASE}/graph`, { headers: headers() });
  return json<GraphSummary>(response);
}

export async function clearGraph(): Promise<void> {
  const response = await fetch(`${BASE}/graph`, {
    method: "DELETE",
    headers: headers(),
  });
  await json(response);
}

export type IndexProgress =
  | { stage: "cloning" | "parsing"; repo: string }
  | { stage: "parsed"; symbols: number; files: number }
  | { stage: "importing"; imported: number; total: number };

export type IndexHandlers = {
  onProgress: (progress: IndexProgress) => void;
  onDone: (imported: number, repo: string) => void;
  onError: (message: string) => void;
};

/** Index a repository, reporting each stage as the server reaches it. */
export function indexRepository(
  url: string,
  branch: string,
  handlers: IndexHandlers,
): () => void {
  const params = new URLSearchParams({ url });
  if (branch) params.set("branch", branch);
  if (API_KEY) params.set("api_key", API_KEY);
  const source = new EventSource(`${BASE}/graph/index?${params}`);

  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.error) {
      handlers.onError(payload.error);
      source.close();
    } else if (payload.done) {
      handlers.onDone(payload.imported, payload.repo);
      source.close();
    } else {
      handlers.onProgress(payload as IndexProgress);
    }
  };
  source.onerror = () => {
    handlers.onError("The connection to the API was lost.");
    source.close();
  };

  return () => source.close();
}

/** Import an index built by the extract CLI - how a local checkout gets in. */
export async function importIndexFile(
  file: File,
): Promise<{ imported: number; repo: string }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${BASE}/graph/import`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  return json<{ imported: number; repo: string }>(response);
}

// ---------------------------------------------------------------------------
// Uploaded PDFs
// ---------------------------------------------------------------------------

export type PdfDocument = {
  id: string;
  name: string;
  chunks: number;
  pages: number;
  uploaded_at: string;
};

export async function listDocuments(): Promise<PdfDocument[]> {
  const response = await fetch(`${BASE}/pdf/documents`, { headers: headers() });
  const body = await json<{ documents: PdfDocument[] }>(response);
  return body.documents;
}

export async function uploadDocument(file: File): Promise<PdfDocument> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${BASE}/pdf/documents`, {
    method: "POST",
    headers: headers(),
    body: form,
  });
  const body = await json<{ document: PdfDocument }>(response);
  return body.document;
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${BASE}/pdf/documents/${id}`, {
    method: "DELETE",
    headers: headers(),
  });
  await json(response);
}

/** Conversations bucketed the way the sidebar shows them. */
export function groupByAge(conversations: Conversation[]): [string, Conversation[]][] {
  const startOfDay = (date: Date) =>
    new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  const today = startOfDay(new Date());
  const day = 86_400_000;
  const buckets = new Map<string, Conversation[]>();

  for (const conversation of conversations) {
    const updated = new Date(conversation.updated_at);
    const age = Number.isNaN(updated.getTime())
      ? Infinity
      : Math.round((today - startOfDay(updated)) / day);
    const heading =
      age <= 0
        ? "Today"
        : age === 1
          ? "Yesterday"
          : age <= 7
            ? "Previous 7 days"
            : age <= 30
              ? "Previous 30 days"
              : "Older";
    buckets.set(heading, [...(buckets.get(heading) ?? []), conversation]);
  }

  const order = ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"];
  return order
    .filter((heading) => buckets.has(heading))
    .map((heading) => [heading, buckets.get(heading)!]);
}
