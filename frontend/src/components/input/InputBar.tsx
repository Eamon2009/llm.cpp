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
    if (!ref.current) return;
    ref.current.style.height = "auto";
    ref.current.style.height = `${Math.min(ref.current.scrollHeight, 120)}px`;
  }, [value]);

  const submit = (): void => {
    const prompt = value.trim();
    if (!prompt || disabled || isSending) return;
    setValue("");
    onSend(prompt);
  };

  return (
    <div
      style={{
        position: "sticky",
        bottom: 0,
        borderTop: "1px solid var(--border-subtle)",
        background: "var(--bg-base)",
        padding: "12px 16px 16px",
      }}
    >
      <div
        style={{
          maxWidth: 780,
          margin: "0 auto",
          borderRadius: 10,
          border: "1px solid var(--border-muted)",
          background: "var(--bg-surface)",
          transition: "border-color 0.15s",
          boxShadow: "0 -8px 32px rgba(0,0,0,0.3)",
        }}
        onFocusCapture={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border-strong)";
        }}
        onBlurCapture={(e) => {
          (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border-muted)";
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-end", gap: 8, padding: "10px 12px 8px" }}>
          <textarea
            ref={ref}
            rows={1}
            value={value}
            disabled={disabled || isSending}
            onChange={(event) => setValue(event.target.value.slice(0, 500))}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder={disabled ? "Server offline — check backend" : "Message Quadtrix.cpp…"}
            style={{
              flex: 1,
              resize: "none",
              background: "transparent",
              border: "none",
              outline: "none",
              color: "var(--text-primary)",
              fontSize: 13,
              lineHeight: 1.6,
              maxHeight: 120,
              minHeight: 22,
              fontFamily: "var(--font-sans)",
            }}
          />
          <button
            disabled={!value.trim() || disabled || isSending}
            onClick={submit}
            type="button"
            style={{
              height: 34,
              minWidth: 70,
              borderRadius: 7,
              border: "none",
              background:
                !value.trim() || disabled || isSending
                  ? "var(--bg-elevated)"
                  : "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
              color:
                !value.trim() || disabled || isSending ? "var(--text-muted)" : "#fff",
              fontSize: 12,
              fontWeight: 600,
              cursor: !value.trim() || disabled || isSending ? "not-allowed" : "pointer",
              transition: "background 0.15s, color 0.15s",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 5,
              flexShrink: 0,
            }}
          >
            {isSending ? (
              <>
                <span
                  style={{
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    border: "1.5px solid rgba(255,255,255,0.3)",
                    borderTopColor: "#fff",
                    animation: "spin 0.7s linear infinite",
                    display: "inline-block",
                  }}
                />
                <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
              </>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 7h11M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
            {isSending ? "Wait" : "Send"}
          </button>
        </div>

        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "0 12px 8px",
          }}
        >
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>
            Enter to send · Shift+Enter for newline
          </span>
          <CharCounter count={value.length} max={500} />
        </div>
      </div>
    </div>
  );
}
