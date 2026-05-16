import { formatDistanceToNow } from "date-fns";

import type { Session } from "../../types";

interface SessionItemProps {
  session: Session;
  active: boolean;
  onSelect: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}

export function SessionItem({ session, active, onSelect, onDelete }: SessionItemProps) {
  return (
    <div
      className={`group flex h-12 items-center gap-2 rounded-md px-2 transition ${
        active ? "bg-[var(--accent-dim)] text-[var(--text-primary)]" : "text-[var(--text-secondary)] hover:bg-hover"
      }`}
    >
      <button className="min-w-0 flex-1 text-left" onClick={() => onSelect(session.id)} type="button">
        <div className="truncate text-sm">{session.title}</div>
        <div className="truncate text-xs text-[var(--text-muted)]">
          {formatDistanceToNow(new Date(session.updated_at), { addSuffix: true })}
        </div>
      </button>
      <button
        aria-label="Delete session"
        className="hidden h-7 w-7 rounded-md text-red-300 hover:bg-red-500/10 group-hover:block"
        onClick={() => onDelete(session.id)}
        type="button"
      >
        x
      </button>
    </div>
  );
}
