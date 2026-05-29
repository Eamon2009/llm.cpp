import type { ReactNode } from "react";

import { SettingsPanel } from "../panels/SettingsPanel";
import { StatsPanel } from "../panels/StatsPanel";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

interface AppLayoutProps {
  children: ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        background: "var(--bg-base)",
        color: "var(--text-primary)",
      }}
    >
      <Sidebar />
      <main style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0 }}>
        <Topbar />
        {children}
      </main>
      <SettingsPanel />
      <StatsPanel />
    </div>
  );
}
