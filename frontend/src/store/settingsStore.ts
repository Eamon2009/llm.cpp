import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { ModelBackend } from "../types";

interface SettingsState {
  apiBaseUrl: string;
  maxTokens: number;
  temperature: number;
  modelLabel: string;
  modelBackend: ModelBackend;
  settingsOpen: boolean;
  statsOpen: boolean;
  setApiBaseUrl: (value: string) => void;
  setMaxTokens: (value: number) => void;
  setTemperature: (value: number) => void;
  setModelLabel: (value: string) => void;
  setModelBackend: (value: ModelBackend) => void;
  setSettingsOpen: (value: boolean) => void;
  setStatsOpen: (value: boolean) => void;
}

const defaultApiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3001";

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      apiBaseUrl: defaultApiBaseUrl,
      maxTokens: 200,
      temperature: 1.0,
      modelLabel: "quadtrix-v1.0 - cpu",
      modelBackend: "cpp",
      settingsOpen: false,
      statsOpen: false,
      setApiBaseUrl: (value) => set({ apiBaseUrl: value }),
      setMaxTokens: (value) => set({ maxTokens: value }),
      setTemperature: (value) => set({ temperature: value }),
      setModelLabel: (value) => set({ modelLabel: value }),
      setModelBackend: (value) => set({ modelBackend: value }),
      setSettingsOpen: (value) => set({ settingsOpen: value }),
      setStatsOpen: (value) => set({ statsOpen: value }),
    }),
    { name: "quadtrix-settings" },
  ),
);
