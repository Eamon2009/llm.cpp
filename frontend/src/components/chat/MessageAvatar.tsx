import type { Role } from "../../types";

interface MessageAvatarProps {
  role: Role;
}

export function MessageAvatar({ role }: MessageAvatarProps) {
  const isUser = role === "user";
  return (
    <div
      className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-md border text-xs font-semibold ${
        isUser
          ? "border-[var(--border-muted)] bg-elevated text-[var(--text-primary)]"
          : "border-[var(--border-muted)] bg-surface font-mono text-[var(--text-primary)]"
      }`}
    >
      {isUser ? "You" : "Q"}
    </div>
  );
}
