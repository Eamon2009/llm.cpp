import { motion } from "framer-motion";

import { useStats } from "../../api/health";
import { useSettingsStore } from "../../store/settingsStore";
import { formatDuration } from "../../utils/time";
import { Button } from "../ui/Button";

export function StatsPanel() {
  const statsOpen = useSettingsStore((state) => state.statsOpen);
  const setStatsOpen = useSettingsStore((state) => state.setStatsOpen);
  const { data, isLoading, isError } = useStats();

  if (!statsOpen) {
    return null;
  }

  const rows = data
    ? [
        ["Model", data.model],
        ["Architecture", data.architecture],
        ["Parameters", data.parameters.toLocaleString()],
        ["Vocabulary", `${data.vocabulary} characters`],
        ["Val Loss", `${data.val_loss} nats`],
        ["Context", `${data.context} characters`],
        ["Training", data.training],
        ["C++ Backend", `${data.backend} ${data.backend_online ? "ok" : "offline"}`],
        ["PyTorch", `${data.torch_checkpoint} ${data.torch_online ? "ok" : "missing"}`],
        ["Uptime", formatDuration(data.uptime_seconds)],
      ]
    : [];

  return (
    <motion.aside
      animate={{ x: 0 }}
      className="fixed right-0 top-0 z-40 h-full w-[264px] border-l border-[var(--border-subtle)] bg-surface p-4 shadow-2xl"
      initial={{ x: 264 }}
      transition={{ duration: 0.18 }}
    >
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">Model Stats</h2>
        <Button className="h-8 px-2" onClick={() => setStatsOpen(false)}>
          x
        </Button>
      </div>
      {isLoading && <div className="text-sm text-[var(--text-secondary)]">Loading stats...</div>}
      {isError && <div className="text-sm text-red-300">Stats unavailable</div>}
      <div className="space-y-3">
        {rows.map(([label, value]) => (
          <div className="flex items-start justify-between gap-3 border-b border-[var(--border-subtle)] pb-2" key={label}>
            <span className="text-xs text-[var(--text-muted)]">{label}</span>
            <span className="text-right text-xs text-[var(--text-primary)]">{value}</span>
          </div>
        ))}
      </div>
    </motion.aside>
  );
}
