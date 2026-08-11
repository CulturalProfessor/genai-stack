import { useEffect, useState } from "react";

import { NavRail, type View } from "@/components/NavRail";
import { ChatView } from "@/views/ChatView";
import { IndexView } from "@/views/IndexView";
import { getGraph } from "@/lib/api";

export default function App() {
  const [view, setView] = useState<View>("chat");
  const [status, setStatus] = useState<string | null>(null);
  const [model, setModel] = useState<string | null>(null);

  // One call at startup tells us the API and Neo4j are both answering.
  useEffect(() => {
    getGraph()
      .then((graph) => {
        setModel(graph.model);
        setStatus(`ok - ${graph.counts.symbols.toLocaleString()} symbols indexed`);
      })
      .catch((exc: Error) => setStatus(`API unreachable: ${exc.message}`));
  }, [view]);

  return (
    <div className="flex h-screen overflow-hidden">
      <NavRail view={view} onChange={setView} status={status} />
      {view === "chat" && <ChatView model={model} />}
      {view === "index" && <IndexView />}
    </div>
  );
}
