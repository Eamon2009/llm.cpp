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
      initial={{ opacity: 0, y: 6 }}
      transition={{ duration: 0.18 }}
      className="group"
      style={{
        display: "flex",
        width: "100%",
        gap: 12,
        justifyContent: isUser ? "flex-end" : "flex-start",
        alignItems: "flex-start",
      }}
    >
      {!isUser && <MessageAvatar role={message.role} />}

      <div
        style={{
          maxWidth: "min(680px, calc(100vw - 80px))",
          display: "flex",
          flexDirection: "column",
          gap: 4,
          alignItems: isUser ? "flex-end" : "flex-start",
        }}
      >
        {/* Meta row */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            fontSize: 11,
            color: "var(--text-muted)",
          }}
        >
          <span style={{ fontWeight: 500 }}>{isUser ? "You" : "Quadtrix"}</span>
          <span>{formatRelativeTime(message.created_at)}</span>
          {!isUser && !message.pending && (
            <button
              className="group-hover:opacity-100"
              onClick={copyText}
              type="button"
              style={{
                opacity: 0,
                background: "none",
                border: "none",
                cursor: "pointer",
                color: copied ? "var(--status-online)" : "var(--text-muted)",
                fontSize: 11,
                padding: "0 2px",
                transition: "opacity 0.12s, color 0.12s",
              }}
            >
              {copied ? "✓ Copied" : "Copy"}
            </button>
          )}
        </div>

        {/* Bubble */}
        <div
          style={{
            borderRadius: 10,
            padding: "10px 14px",
            fontSize: 13,
            lineHeight: 1.7,
            ...(isUser
              ? {
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-muted)",
                  color: "var(--text-primary)",
                }
              : message.error
              ? {
                  background: "rgba(224,82,82,0.08)",
                  border: "1px solid rgba(224,82,82,0.2)",
                  color: "#f87171",
                }
              : {
                  background: "var(--bg-surface)",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }),
          }}
        >
          {message.pending ? (
            <ThinkingIndicator />
          ) : (
            <span style={{ whiteSpace: "pre-wrap" }}>{message.text}</span>
          )}
        </div>
      </div>

      {isUser && <MessageAvatar role={message.role} />}
    </motion.div>
  );
}
