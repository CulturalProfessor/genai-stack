import { useState } from "react";
import {
  MessageSquare,
  MoreHorizontal,
  Paperclip,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { groupByAge, type Conversation } from "@/lib/api";

type Props = {
  title: string;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  footer?: React.ReactNode;
};

export function Sidebar({
  title,
  conversations,
  activeId,
  onSelect,
  onNew,
  onRename,
  onDelete,
  footer,
}: Props) {
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  const startRename = (conversation: Conversation) => {
    setRenaming(conversation.id);
    setDraft(conversation.title);
  };

  const commitRename = () => {
    if (renaming && draft.trim()) onRename(renaming, draft.trim());
    setRenaming(null);
  };

  return (
    <aside className="flex h-full w-[17rem] shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
      <div className="p-3">
        <p className="mb-3 px-1 text-sm font-semibold tracking-tight">{title}</p>
        <Button onClick={onNew} className="w-full gap-2">
          <Plus /> New chat
        </Button>
      </div>

      <nav className="flex-1 overflow-y-auto px-2 pb-2">
        {conversations.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-muted-foreground">
            No chats yet.
          </p>
        )}

        {groupByAge(conversations).map(([heading, bucket]) => (
          <div key={heading} className="mb-1">
            <p className="px-2 pb-1 pt-3 text-[0.65rem] font-semibold uppercase tracking-wider text-muted-foreground/70">
              {heading}
            </p>
            {bucket.map((conversation) =>
              renaming === conversation.id ? (
                <div key={conversation.id} className="px-1 py-0.5">
                  <Input
                    autoFocus
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") commitRename();
                      if (event.key === "Escape") setRenaming(null);
                    }}
                    className="h-8 text-sm"
                  />
                </div>
              ) : (
                <div
                  key={conversation.id}
                  className={cn(
                    "group flex items-center gap-1 rounded-md px-1",
                    activeId === conversation.id
                      ? "bg-sidebar-accent"
                      : "hover:bg-white/[0.05]",
                  )}
                >
                  <button
                    onClick={() => onSelect(conversation.id)}
                    title={conversation.title}
                    className="flex min-w-0 flex-1 items-center gap-2 py-2 pl-1 text-left text-sm"
                  >
                    <MessageSquare className="size-3.5 shrink-0 text-muted-foreground" />
                    <span className="truncate">{conversation.title}</span>
                    {conversation.document_count > 0 && (
                      <Paperclip
                        className="size-3 shrink-0 text-muted-foreground"
                        aria-label={`${conversation.document_count} attached`}
                      />
                    )}
                  </button>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-7 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 data-[state=open]:opacity-100"
                        aria-label="Chat actions"
                      >
                        <MoreHorizontal />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onSelect={() => startRename(conversation)}>
                        <Pencil className="size-3.5" /> Rename
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        destructive
                        onSelect={() => onDelete(conversation.id)}
                      >
                        <Trash2 className="size-3.5" /> Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              ),
            )}
          </div>
        ))}
      </nav>

      <div className="space-y-1 border-t border-sidebar-border px-4 py-3 text-[0.7rem] text-muted-foreground">
        {footer}
        <a
          href="http://localhost:7474"
          target="_blank"
          rel="noreferrer"
          className="block hover:text-foreground"
        >
          Browse the graph ↗
        </a>
      </div>
    </aside>
  );
}
