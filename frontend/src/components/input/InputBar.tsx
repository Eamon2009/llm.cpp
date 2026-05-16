import { useEffect, useRef, useState } from "react";

import { CharCounter } from "./CharCounter";

interface InputBarProps {
  disabled: boolean;
  isSending: boolean;
  onSend: (prompt: string) => void;
  initialValue?: string;
  onDraftChange?: (value: string) => void;
}

export function InputBar({ disabled, isSending, onSend, initialValue = "", onDraftChange }: InputBarProps) {
  const [value, setValue] = useState(initialValue);
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    setValue(initialValue);
  }, [initialValue]);

  useEffect(() => {
    onDraftChange?.(value);
  }, [onDraftChange, value]);

  useEffect(() => {
    if (!ref.current) {
      return;
    }
    ref.current.style.height = "auto";
    ref.current.style.height = `${Math.min(ref.current.scrollHeight, 120)}px`;
  }, [value]);

  const submit = (): void => {
    const prompt = value.trim();
    if (!prompt || disabled || isSending) {
      return;
    }
    setValue("");
    onSend(prompt);
  };

  return (
    <div className="sticky bottom-0 border-t border-[var(--border-subtle)] bg-base px-4 py-4 md:px-8">
      <div className="mx-auto max-w-4xl rounded-lg border border-[var(--border-muted)] bg-input p-2 shadow-[0_-16px_40px_rgba(0,0,0,0.35)] focus-within:border-[var(--border-strong)]">
        <div className="flex items-end gap-2">
          <textarea
            className="max-h-[120px] min-h-10 flex-1 resize-none bg-transparent px-2 py-2 font-mono text-sm leading-6 text-[var(--text-primary)] outline-none placeholder:text-[var(--text-muted)]"
            disabled={disabled || isSending}
            onChange={(event) => setValue(event.target.value.slice(0, 500))}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={disabled ? "Server offline" : "Message Quadtrix.cpp"}
            ref={ref}
            rows={1}
            value={value}
          />
          <button
            className="h-9 min-w-20 rounded-md border border-[var(--border-strong)] bg-[var(--text-primary)] px-3 text-sm font-medium text-black transition hover:bg-white disabled:border-[var(--border-muted)] disabled:bg-[var(--bg-hover)] disabled:text-[var(--text-muted)]"
            disabled={!value.trim() || disabled || isSending}
            onClick={submit}
            type="button"
          >
            {isSending ? "Thinking..." : "Send"}
          </button>
        </div>
        <div className="mt-1 flex items-center justify-between px-2">
          <span className="text-xs text-[var(--text-muted)]">Enter to send - Shift+Enter newline</span>
          <CharCounter count={value.length} max={500} />
        </div>
      </div>
    </div>
  );
}
