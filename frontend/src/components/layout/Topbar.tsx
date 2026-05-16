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
      : "PyTorch checkpoint - engine/best_model .pt";
  return (
    <header className="border-b border-[var(--border-subtle)] bg-base/95 px-4 py-3 backdrop-blur md:px-6">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-md border border-[var(--border-muted)] bg-white">
            <img alt="Quadtrix.cpp icon" className="h-8 w-8 object-contain" src="/icon.svg" />
          </div>
          <div className="min-w-0">
            <div className="truncate font-mono text-sm font-semibold tracking-[0.18em] text-[var(--text-primary)]">
              Quadtrix.cpp
            </div>
          </div>
        </div>
        <div className="hidden items-center gap-2 md:flex">
          <ModelBadge degraded={Boolean(data && !online)} label={badgeLabel} online={online && !isError} tooltip={tooltip} />
          <select
            className="h-9 rounded-md border border-[var(--border-muted)] bg-input px-2 text-sm text-[var(--text-primary)] outline-none transition hover:border-[var(--border-strong)]"
            onChange={(event) => setModelBackend(event.target.value as ModelBackend)}
            value={modelBackend}
          >
            <option value="cpp">C++</option>
            <option value="torch">.pt</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={clearMessages}>Clear</Button>
          <Button onClick={() => setStatsOpen(true)}>Stats</Button>
          <Button onClick={() => setSettingsOpen(true)}>Settings</Button>
        </div>
      </div>
      <div className="mx-auto mt-3 flex max-w-6xl items-center gap-2 md:hidden">
        <ModelBadge degraded={Boolean(data && !online)} label={badgeLabel} online={online && !isError} tooltip={tooltip} />
        <select
          className="h-9 rounded-md border border-[var(--border-muted)] bg-input px-2 text-sm text-[var(--text-primary)] outline-none transition hover:border-[var(--border-strong)]"
          onChange={(event) => setModelBackend(event.target.value as ModelBackend)}
          value={modelBackend}
        >
          <option value="cpp">C++</option>
          <option value="torch">.pt</option>
        </select>
      </div>
    </header>
  );
}
