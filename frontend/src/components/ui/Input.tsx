import type { InputHTMLAttributes } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export function Input({ label, className = "", ...props }: InputProps) {
  const input = (
    <input
      className={`h-10 w-full rounded-md border border-[var(--border-muted)] bg-input px-3 text-sm text-[var(--text-primary)] outline-none transition placeholder:text-[var(--text-muted)] focus:border-[var(--border-strong)] ${className}`}
      {...props}
    />
  );
  if (!label) {
    return input;
  }
  return (
    <label className="block space-y-2 text-sm text-[var(--text-secondary)]">
      <span>{label}</span>
      {input}
    </label>
  );
}
