import { useEffect, useState } from "react";

import { ApiClientError } from "../../api/client";
import { useSendMessage } from "../../api/chat";
import { useSessionMessages } from "../../api/sessions";
import { useHealth } from "../../api/health";
import { useSessionStore } from "../../store/sessionStore";
import { useSettingsStore } from "../../store/settingsStore";
import type { Message } from "../../types";
import { EmptyState } from "./EmptyState";
import { MessageList } from "./MessageList";
import { InputBar } from "../input/InputBar";

export function ChatView() {
  const [draft, setDraft] = useState("");
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const messages = useSessionStore((state) => state.messages);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const setMessages = useSessionStore((state) => state.setMessages);
  const addMessage = useSessionStore((state) => state.addMessage);
  const replaceMessage = useSessionStore((state) => state.replaceMessage);
  const maxTokens = useSettingsStore((state) => state.maxTokens);
  const temperature = useSettingsStore((state) => state.temperature);
  const modelBackend = useSettingsStore((state) => state.modelBackend);
  const sendMessage = useSendMessage();
  const { data: health } = useHealth();
  const { data: fetchedMessages } = useSessionMessages(activeSessionId);
  const online = modelBackend === "cpp" ? health?.cpp_server === "ok" : health?.torch_model === "ok";

  useEffect(() => {
    if (fetchedMessages) {
      setMessages(fetchedMessages);
    }
  }, [fetchedMessages, setMessages]);

  const now = (): string => new Date().toISOString();

  const handleSend = (prompt: string): void => {
    const sessionId = activeSessionId ?? crypto.randomUUID();
    setActiveSession(sessionId);
    const userMessage: Message = {
      id: `local-${crypto.randomUUID()}`,
      session_id: sessionId,
      role: "user",
      text: prompt,
      chars: 0,
      seconds: 0,
      created_at: now(),
    };
    const pendingId = `pending-${crypto.randomUUID()}`;
    const pendingMessage: Message = {
      id: pendingId,
      session_id: sessionId,
      role: "assistant",
      text: "",
      chars: 0,
      seconds: 0,
      created_at: now(),
      pending: true,
    };
    addMessage(userMessage);
    addMessage(pendingMessage);
    sendMessage.mutate(
      { session_id: sessionId, prompt, max_tokens: maxTokens, temperature, stream: false, model_backend: modelBackend },
      {
        onSuccess: (response) => {
          setActiveSession(response.session_id);
          replaceMessage(pendingId, {
            id: response.id,
            session_id: response.session_id,
            role: "assistant",
            text: response.text,
            prompt: response.prompt,
            chars: response.chars,
            seconds: response.seconds,
            created_at: response.created_at,
          });
        },
        onError: (error) => {
          const message =
            error instanceof ApiClientError
              ? error.apiError.message
              : "Could not reach the selected model. Check the C++ server or engine checkpoint.";
          replaceMessage(pendingId, {
            ...pendingMessage,
            text: message,
            pending: false,
            error: "model_unavailable",
          });
        },
      },
    );
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-base">
      {messages.length === 0 ? <EmptyState /> : <MessageList messages={messages} />}
      <InputBar
        disabled={!online}
        initialValue={draft}
        isSending={sendMessage.isPending}
        onDraftChange={setDraft}
        onSend={handleSend}
      />
    </div>
  );
}
