import type { InputHTMLAttributes } from "react";

interface SliderProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  valueLabel: string;
}

export function Slider({ label, valueLabel, className = "", ...props }: SliderProps) {
  return (
    <label className={`block space-y-2 ${className}`}>
      <span className="flex items-center justify-between text-sm text-[var(--text-secondary)]">
        <span>{label}</span>
        <span className="font-mono text-[var(--text-primary)]">{valueLabel}</span>
      </span>
      <input className="h-2 w-full accent-[var(--accent)]" type="range" {...props} />
    </label>
  );
}
