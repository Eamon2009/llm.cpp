interface StarterPromptsProps {
  onSelect: (prompt: string) => void;
}

const prompts = ["Once upon a time", "Timmy is a", "hi how are you", "The little door opened"];

export function StarterPrompts({ onSelect }: StarterPromptsProps) {
  return (
    <div className="grid w-full max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
      {prompts.map((prompt) => (
        <button
          className="rounded-md border border-[var(--border-muted)] bg-surface px-4 py-3 text-left text-sm text-[var(--text-secondary)] transition hover:border-[var(--border-strong)] hover:bg-hover hover:text-[var(--text-primary)]"
          key={prompt}
          onClick={() => onSelect(prompt)}
          type="button"
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}
