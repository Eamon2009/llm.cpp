import { create } from "zustand";

import type { Message, Session } from "../types";

interface SessionState {
  sessions: Session[];
  activeSessionId: string | null;
  messages: Message[];
  setSessions: (sessions: Session[]) => void;
  setActiveSession: (sessionId: string | null) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  replaceMessage: (id: string, message: Message) => void;
  clearMessages: () => void;
  upsertSession: (session: Session) => void;
  removeSession: (sessionId: string) => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  sessions: [],
  activeSessionId: null,
  messages: [],
  setSessions: (sessions) => set({ sessions }),
  setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  replaceMessage: (id, message) =>
    set((state) => ({ messages: state.messages.map((item) => (item.id === id ? message : item)) })),
  clearMessages: () => set({ messages: [], activeSessionId: null }),
  upsertSession: (session) =>
    set((state) => {
      const exists = state.sessions.some((item) => item.id === session.id);
      const sessions = exists
        ? state.sessions.map((item) => (item.id === session.id ? session : item))
        : [session, ...state.sessions];
      return { sessions: sessions.slice(0, 50) };
    }),
  removeSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((item) => item.id !== sessionId),
      activeSessionId: state.activeSessionId === sessionId ? null : state.activeSessionId,
      messages: state.activeSessionId === sessionId ? [] : state.messages,
    })),
}));
