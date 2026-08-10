import { ChevronLeft, ChevronRight, Layers3 } from "lucide-react";
import type { Destination, Revision } from "../types";
import { AudioCard } from "./AudioCard";

interface Props {
  revisions: Revision[];
  expanded: boolean;
  onExpandedChange: (value: boolean) => void;
  onRoute: (source: string, destination: Destination) => void;
}

export function HistoryDrawer({ revisions, expanded, onExpandedChange, onRoute }: Props) {
  return (
    <aside className={`history-drawer ${expanded ? "history-drawer--expanded" : ""}`}>
      <header className="history-header">
        {expanded && <><Layers3 size={16} /><strong>Revision History</strong></>}
        <button
          className="icon-button"
          aria-label={expanded ? "Collapse history" : "Expand history"}
          onClick={() => onExpandedChange(!expanded)}
        >
          {expanded ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}
        </button>
      </header>
      <div className="history-list">
        {revisions.map((revision, index) => (
          <AudioCard
            key={revision.id}
            title={expanded ? `${revision.metadata.kind === "edit" ? "Edit" : "TTS"} · ${new Date(revision.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : `R${revisions.length - index}`}
            data={{
              sourceId: revision.id,
              audioUrl: revision.audio_url,
              text: revision.text,
            }}
            compact={!expanded}
            showTranscript={expanded}
            onRoute={onRoute}
          />
        ))}
        {!revisions.length && <div className="history-empty">{expanded ? "No revisions" : "—"}</div>}
      </div>
    </aside>
  );
}
