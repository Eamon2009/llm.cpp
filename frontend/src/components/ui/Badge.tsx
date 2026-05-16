import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  tone?: "default" | "green" | "yellow" | "red";
}

export function Badge({ children, tone = "default" }: BadgeProps) {
  const tones = {
    default: "border-[var(--border-muted)] text-[var(--text-secondary)]",
    green: "border-emerald-500/30 text-emerald-300",
    yellow: "border-yellow-500/30 text-yellow-300",
    red: "border-red-500/30 text-red-300",
  };
  return <span className={`rounded-md border px-2 py-1 text-xs ${tones[tone]}`}>{children}</span>;
}
