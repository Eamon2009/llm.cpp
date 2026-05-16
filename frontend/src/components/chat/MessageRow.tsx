import { motion } from "framer-motion";
import { useState } from "react";

import { formatRelativeTime } from "../../utils/time";
import type { Message } from "../../types";
import { MessageAvatar } from "./MessageAvatar";
import { ThinkingIndicator } from "./ThinkingIndicator";

interface MessageRowProps {
  message: Message;
}

export function MessageRow({ message }: MessageRowProps) {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copyText = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch (error) {
      setCopied(false);
    }
  };

  return (
    <motion.div
      animate={{ opacity: 1, y: 0 }}
      className={`group flex w-full gap-3 ${isUser ? "justify-end" : "justify-start"}`}
      initial={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.2 }}
    >
      {!isUser && <MessageAvatar role={message.role} />}
      <div className={`max-w-[min(760px,calc(100vw-48px))] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-[var(--text-muted)]">
          <span>{isUser ? "You" : "Quadtrix"}</span>
          <span>{formatRelativeTime(message.created_at)}</span>
          {!isUser && !message.pending && (
            <button
              className="hidden rounded px-1 text-[var(--text-secondary)] hover:text-[var(--text-primary)] group-hover:inline"
              onClick={copyText}
              type="button"
            >
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>
        <div
          className={`rounded-lg border px-4 py-3 text-sm leading-7 ${
            isUser
              ? "border-[var(--border-muted)] bg-surface text-[var(--text-primary)]"
              : message.error
                ? "border-red-500/20 bg-red-500/10 font-sans text-red-200"
                : "border-[var(--border-subtle)] bg-[#0d0d0d] font-mono text-[var(--text-primary)]"
          }`}
        >
          {message.pending ? <ThinkingIndicator /> : <span className="whitespace-pre-wrap">{message.text}</span>}
        </div>
      </div>
      {isUser && <MessageAvatar role={message.role} />}
    </motion.div>
  );
}
