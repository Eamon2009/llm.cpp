export function EmptyState() {
  return (
    <div className="flex flex-1 items-center justify-center px-6">
      <div className="flex w-full max-w-3xl flex-col items-center gap-6 text-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-md border border-[var(--border-muted)] bg-white">
          <img alt="Quadtrix.cpp icon" className="h-14 w-14 object-contain" src="/icon.svg" />
        </div>
        <div className="space-y-2">
          <h1 className="font-mono text-2xl font-semibold tracking-[0.18em] text-[var(--text-primary)]">Quadtrix.cpp</h1>
          <p className="text-sm text-[var(--text-secondary)]">Minimal local chat interface. Start typing below to begin.</p>
        </div>
      </div>
    </div>
  );
}
