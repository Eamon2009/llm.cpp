import type { ReactNode } from "react";

import { SettingsPanel } from "../panels/SettingsPanel";
import { StatsPanel } from "../panels/StatsPanel";
import { Topbar } from "./Topbar";

interface AppLayoutProps {
  children: ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden bg-base text-[var(--text-primary)]">
      <main className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        {children}
      </main>
      <SettingsPanel />
      <StatsPanel />
    </div>
  );
}
