import { useEffect } from "react";

import { useCreateSession, useDeleteSession, useSessionMessages, useSessions } from "../../api/sessions";
import { useSessionStore } from "../../store/sessionStore";
import { NewChatButton } from "../sidebar/NewChatButton";
import { SessionList } from "../sidebar/SessionList";

export function Sidebar() {
  const sessions = useSessionStore((state) => state.sessions);
  const activeSessionId = useSessionStore((state) => state.activeSessionId);
  const setSessions = useSessionStore((state) => state.setSessions);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const setMessages = useSessionStore((state) => state.setMessages);
  const removeSession = useSessionStore((state) => state.removeSession);
  const sessionsQuery = useSessions();
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();
  const messagesQuery = useSessionMessages(activeSessionId);

  useEffect(() => {
    if (sessionsQuery.data) {
      setSessions(sessionsQuery.data);
    }
  }, [sessionsQuery.data, setSessions]);

  useEffect(() => {
    if (messagesQuery.data) {
      setMessages(messagesQuery.data);
    }
  }, [messagesQuery.data, setMessages]);

  const newConversation = (): void => {
    createSession.mutate(undefined, {
      onSuccess: (session) => {
        setActiveSession(session.id);
        setMessages([]);
      },
    });
  };

  const deleteConversation = (sessionId: string): void => {
    deleteSession.mutate(sessionId, {
      onSuccess: () => removeSession(sessionId),
    });
  };

  return (
    <aside className="hidden w-[264px] shrink-0 flex-col border-r border-[var(--border-subtle)] bg-surface p-3 md:flex">
      <div className="mb-5 flex items-center gap-3 px-2 py-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-md bg-accent font-bold text-white">Q</div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-[var(--text-primary)]">Quadtrix.cpp</div>
          <div className="truncate text-xs text-[var(--text-muted)]">char-level GPT v1.0</div>
        </div>
      </div>
      <NewChatButton onClick={newConversation} />
      <div className="mt-6 text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">Recent</div>
      <div className="mt-2 min-h-0 flex-1">
        <SessionList
          activeSessionId={activeSessionId}
          onDelete={deleteConversation}
          onSelect={setActiveSession}
          sessions={sessions}
        />
      </div>
      <div className="mt-4 border-t border-[var(--border-subtle)] pt-4 text-xs text-[var(--text-muted)]">
        API Configuration
      </div>
    </aside>
  );
}
