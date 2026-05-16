import { Tooltip } from "../ui/Tooltip";

interface ModelBadgeProps {
  online: boolean;
  degraded: boolean;
  label: string;
  tooltip: string;
}

export function ModelBadge({ online, degraded, label, tooltip }: ModelBadgeProps) {
  const tone = online ? "bg-[var(--status-online)]" : degraded ? "animate-pulse bg-[var(--yellow)]" : "bg-[var(--red)]";
  return (
    <Tooltip label={tooltip}>
      <div className="flex h-9 items-center gap-2 rounded-md border border-[var(--border-muted)] bg-surface px-3 text-sm text-[var(--text-secondary)]">
        <span className={`h-2 w-2 rounded-full ${tone}`} />
        <span className="max-w-48 truncate">{label}</span>
      </div>
    </Tooltip>
  );
}
