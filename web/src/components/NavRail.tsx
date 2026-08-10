import { FileText, MessagesSquare, Network } from "lucide-react";

import { cn } from "@/lib/utils";

export type View = "chat" | "index" | "pdf";

const ITEMS: { id: View; label: string; icon: typeof Network }[] = [
  { id: "chat", label: "Ask the codebase", icon: MessagesSquare },
  { id: "pdf", label: "Chat with a PDF", icon: FileText },
  { id: "index", label: "Index a repository", icon: Network },
];

export function NavRail({
  view,
  onChange,
  status,
}: {
  view: View;
  onChange: (view: View) => void;
  status: string | null;
}) {
  return (
    <div className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-sidebar-border bg-sidebar py-3">
      <div className="mb-2 text-lg" title="Code graph">
        🕸️
      </div>

      {ITEMS.map((item) => (
        <button
          key={item.id}
          onClick={() => onChange(item.id)}
          title={item.label}
          aria-label={item.label}
          aria-current={view === item.id}
          className={cn(
            "flex size-10 items-center justify-center rounded-xl transition-colors",
            view === item.id
              ? "bg-primary/20 text-primary"
              : "text-muted-foreground hover:bg-white/[0.06] hover:text-foreground",
          )}
        >
          <item.icon className="size-[1.15rem]" />
        </button>
      ))}

      <div className="mt-auto" title={status ?? "checking..."}>
        <span className="text-xs">{status?.startsWith("ok") ? "🟢" : "🔴"}</span>
      </div>
    </div>
  );
}
