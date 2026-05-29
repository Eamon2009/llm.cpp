import { useHealth } from "../../api/health";
import { useSessionStore } from "../../store/sessionStore";
import { useSettingsStore } from "../../store/settingsStore";
import type { ModelBackend } from "../../types";
import { ModelBadge } from "../panels/ModelBadge";
import { Button } from "../ui/Button";

export function Topbar() {
  const { data, isError } = useHealth();
  const clearMessages = useSessionStore((state) => state.clearMessages);
  const setSettingsOpen = useSettingsStore((state) => state.setSettingsOpen);
  const setStatsOpen = useSettingsStore((state) => state.setStatsOpen);
  const modelBackend = useSettingsStore((state) => state.modelBackend);
  const setModelBackend = useSettingsStore((state) => state.setModelBackend);
  const cppOnline = data?.cpp_server === "ok";
  const torchOnline = data?.torch_model === "ok";
  const online = modelBackend === "cpp" ? cppOnline : torchOnline;
  const badgeLabel = modelBackend === "cpp" ? "Quadtrix C++" : "Quadtrix .pt";
  const tooltip =
    modelBackend === "cpp"
      ? "C++ server - run: ./Quadtrix.exe --server --port 8080"
      : "PyTorch checkpoint - engine/best_model.pt";

  return (
    <header
      style={{
        borderBottom: "1px solid var(--border-subtle)",
        background: "rgba(17,17,17,0.97)",
        backdropFilter: "blur(8px)",
        padding: "0 16px",
        height: 52,
        display: "flex",
        alignItems: "center",
        flexShrink: 0,
        gap: 12,
      }}
    >
      {/* Left: title (mobile only, sidebar hidden on mobile) */}
      <div className="md:hidden" style={{ display: "flex", alignItems: "center", gap: 8, marginRight: "auto" }}>
        <div
          style={{
            width: 26,
            height: 26,
            borderRadius: 6,
            background: "linear-gradient(135deg, #4f8ef7 0%, #2563eb 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 700,
            fontSize: 12,
            color: "#fff",
          }}
        >
          Q
        </div>
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)" }}>Quadtrix.cpp</span>
      </div>

      {/* Center stretch spacer on desktop */}
      <div className="hidden md:block" style={{ flex: 1 }} />

      {/* Model selector row */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <ModelBadge degraded={Boolean(data && !online)} label={badgeLabel} online={online && !isError} tooltip={tooltip} />
        <select
          style={{
            height: 32,
            borderRadius: 6,
            border: "1px solid var(--border-muted)",
            background: "var(--bg-elevated)",
            color: "var(--text-primary)",
            padding: "0 8px",
            fontSize: 12,
            fontWeight: 500,
            outline: "none",
            cursor: "pointer",
          }}
          onChange={(event) => setModelBackend(event.target.value as ModelBackend)}
          value={modelBackend}
        >
          <option value="cpp">C++</option>
          <option value="torch">.pt</option>
        </select>
      </div>

      {/* Actions */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <Button onClick={clearMessages}>Clear</Button>
        <Button onClick={() => setStatsOpen(true)}>Stats</Button>
        <Button onClick={() => setSettingsOpen(true)}>Settings</Button>
      </div>
    </header>
  );
}
