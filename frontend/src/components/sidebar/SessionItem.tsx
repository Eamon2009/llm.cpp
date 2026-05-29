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
      className="group"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 4,
        borderRadius: 6,
        padding: "0 4px",
        background: active ? "var(--accent-dim)" : "transparent",
        transition: "background 0.12s",
        marginBottom: 1,
      }}
      onMouseEnter={(e) => {
        if (!active) (e.currentTarget as HTMLDivElement).style.background = "var(--bg-hover)";
      }}
      onMouseLeave={(e) => {
        if (!active) (e.currentTarget as HTMLDivElement).style.background = "transparent";
      }}
    >
      <button
        onClick={() => onSelect(session.id)}
        type="button"
        style={{
          flex: 1,
          minWidth: 0,
          textAlign: "left",
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "8px 4px",
          color: active ? "var(--text-primary)" : "var(--text-secondary)",
        }}
      >
        <div
          style={{
            fontSize: 12,
            fontWeight: active ? 500 : 400,
            color: active ? "var(--text-primary)" : "var(--text-secondary)",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}
        >
          {session.title}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
          {formatDistanceToNow(new Date(session.updated_at), { addSuffix: true })}
        </div>
      </button>
      <button
        aria-label="Delete session"
        onClick={() => onDelete(session.id)}
        type="button"
        className="group-hover:opacity-100"
        style={{
          opacity: 0,
          width: 24,
          height: 24,
          borderRadius: 4,
          border: "none",
          background: "none",
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--red)",
          flexShrink: 0,
          transition: "opacity 0.12s, background 0.12s",
          padding: 0,
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = "rgba(224,82,82,0.12)";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = "none";
        }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M1.5 1.5l9 9M10.5 1.5l-9 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
        </svg>
      </button>
    </div>
  );
}
