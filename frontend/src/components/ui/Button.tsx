import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "danger";
  children: ReactNode;
}

export function Button({ variant = "ghost", className = "", children, ...props }: ButtonProps) {
  const variants = {
    primary: "border border-[var(--border-strong)] bg-[var(--text-primary)] text-black hover:bg-white disabled:border-[var(--border-muted)] disabled:bg-[var(--bg-hover)] disabled:text-[var(--text-muted)]",
    ghost: "border border-[var(--border-muted)] bg-transparent text-[var(--text-primary)] hover:border-[var(--border-strong)] hover:bg-hover",
    danger: "border border-red-500/30 text-red-300 hover:bg-red-500/10",
  };
  return (
    <button
      className={`inline-flex h-9 items-center justify-center rounded-md px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
