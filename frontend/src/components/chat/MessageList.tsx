import { useAutoScroll } from "../../hooks/useAutoScroll";
import type { Message } from "../../types";
import { MessageRow } from "./MessageRow";

interface MessageListProps {
  messages: Message[];
}

export function MessageList({ messages }: MessageListProps) {
  const scrollRef = useAutoScroll<HTMLDivElement>(messages.length);
  return (
    <div className="flex-1 overflow-y-auto px-4 py-8 md:px-8 md:py-10" ref={scrollRef}>
      <div className="mx-auto flex max-w-4xl flex-col gap-8">
        {messages.map((message) => (
          <MessageRow key={message.id} message={message} />
        ))}
      </div>
    </div>
  );
}
