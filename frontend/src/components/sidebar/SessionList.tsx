import type { Session } from "../../types";
import { SessionItem } from "./SessionItem";

interface SessionListProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}

export function SessionList({ sessions, activeSessionId, onSelect, onDelete }: SessionListProps) {
  if (sessions.length === 0) {
    return <div className="px-2 py-3 text-sm text-[var(--text-muted)]">No sessions yet</div>;
  }
  return (
    <div className="space-y-1 overflow-y-auto pr-1">
      {sessions.slice(0, 50).map((session) => (
        <SessionItem
          active={session.id === activeSessionId}
          key={session.id}
          onDelete={onDelete}
          onSelect={onSelect}
          session={session}
        />
      ))}
    </div>
  );
}
