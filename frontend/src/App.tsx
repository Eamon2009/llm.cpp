import { useCallback } from "react";

import { ChatView } from "./components/chat/ChatView";
import { AppLayout } from "./components/layout/AppLayout";
import { useKeyboardShortcut } from "./hooks/useKeyboardShortcut";
import { useCreateSession } from "./api/sessions";
import { useSessionStore } from "./store/sessionStore";
import { useSettingsStore } from "./store/settingsStore";

export default function App() {
  const clearMessages = useSessionStore((state) => state.clearMessages);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const setMessages = useSessionStore((state) => state.setMessages);
  const setSettingsOpen = useSettingsStore((state) => state.setSettingsOpen);
  const createSession = useCreateSession();

  const newConversation = useCallback(() => {
    createSession.mutate(undefined, {
      onSuccess: (session) => {
        setActiveSession(session.id);
        setMessages([]);
      },
    });
  }, [createSession, setActiveSession, setMessages]);

  const openSettings = useCallback(() => setSettingsOpen(true), [setSettingsOpen]);
  const closeSettings = useCallback(() => setSettingsOpen(false), [setSettingsOpen]);

  useKeyboardShortcut(["ctrl", "l"], clearMessages);
  useKeyboardShortcut(["ctrl", "n"], newConversation);
  useKeyboardShortcut(["ctrl", ","], openSettings);
  useKeyboardShortcut(["escape"], closeSettings);

  return (
    <AppLayout>
      <ChatView />
    </AppLayout>
  );
}
