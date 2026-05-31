import type { Role } from "../../types";

interface MessageAvatarProps {
  role: Role;
}

export function MessageAvatar({ role }: MessageAvatarProps) {
  const isUser = role === "user";

  if (isUser) {
    return (
      <div
        style={{
          width: 30,
          height: 30,
          borderRadius: "50%",
          background: "var(--bg-elevated)",
          border: "1px solid var(--border-muted)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 11,
          fontWeight: 600,
          color: "var(--text-secondary)",
          flexShrink: 0,
        }}
      >
        U
      </div>
    );
  }

  return (
    <div
      style={{
        width: 30,
        height: 30,
        borderRadius: "50%",
        background: "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 12,
        fontWeight: 700,
        color: "#fff",
        flexShrink: 0,
        boxShadow: "0 2px 8px rgba(79,142,247,0.3)",
      }}
    >
      Q
    </div>
  );
}
