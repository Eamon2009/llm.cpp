interface CharCounterProps {
  count: number;
  max: number;
}

export function CharCounter({ count, max }: CharCounterProps) {
  const over = count > max;
  return <span className={`text-xs ${over ? "text-red-300" : "text-[var(--text-muted)]"}`}>{count}/{max}</span>;
}
