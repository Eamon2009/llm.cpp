import type { ReactNode } from "react";

interface TooltipProps {
  label: string;
  children: ReactNode;
}

export function Tooltip({ label, children }: TooltipProps) {
  return (
    <span className="group relative inline-flex">
      {children}
      <span className="pointer-events-none absolute right-0 top-full z-30 mt-2 hidden w-max max-w-64 rounded-md border border-[var(--border-muted)] bg-elevated px-2 py-1 text-xs text-[var(--text-secondary)] shadow-xl group-hover:block">
        {label}
      </span>
    </span>
  );
}
