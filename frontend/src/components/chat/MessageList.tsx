import { useRef } from "react";
import type { Message } from "../../types";
import { useAutoScroll } from "../../hooks/useAutoScroll";
import { MessageRow } from "./MessageRow";

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  useAutoScroll(bottomRef, messages);

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "24px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 20,
      }}
    >
      <div style={{ maxWidth: 780, width: "100%", margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
        {messages.map((message) => (
          <MessageRow key={message.id} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
