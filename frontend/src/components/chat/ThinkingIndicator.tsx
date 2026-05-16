export function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 text-[var(--text-secondary)]">
      <span>Quadtrix is thinking</span>
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-secondary)]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-secondary)] [animation-delay:120ms]" />
        <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-[var(--text-secondary)] [animation-delay:240ms]" />
      </span>
    </div>
  );
}
