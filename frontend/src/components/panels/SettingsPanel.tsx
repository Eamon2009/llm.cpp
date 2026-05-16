import { motion } from "framer-motion";

import { useSettingsStore } from "../../store/settingsStore";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { Slider } from "../ui/Slider";

export function SettingsPanel() {
  const {
    apiBaseUrl,
    maxTokens,
    temperature,
    modelLabel,
    modelBackend,
    settingsOpen,
    setApiBaseUrl,
    setMaxTokens,
    setTemperature,
    setModelLabel,
    setModelBackend,
    setSettingsOpen,
  } = useSettingsStore();

  if (!settingsOpen) {
    return null;
  }

  return (
    <motion.aside
      animate={{ x: 0 }}
      className="fixed right-0 top-0 z-40 h-full w-[264px] border-l border-[var(--border-subtle)] bg-surface p-4 shadow-2xl"
      initial={{ x: 264 }}
      transition={{ duration: 0.18 }}
    >
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">Settings</h2>
        <Button className="h-8 px-2" onClick={() => setSettingsOpen(false)}>
          x
        </Button>
      </div>
      <div className="space-y-5">
        <Input label="Backend URL" onChange={(event) => setApiBaseUrl(event.target.value)} value={apiBaseUrl} />
        <label className="block space-y-2 text-sm text-[var(--text-secondary)]">
          <span>Active model</span>
          <select
            className="h-10 w-full rounded-md border border-[var(--border-muted)] bg-input px-3 text-sm text-[var(--text-primary)] outline-none"
            onChange={(event) => setModelBackend(event.target.value === "torch" ? "torch" : "cpp")}
            value={modelBackend}
          >
            <option value="cpp">C++ server</option>
            <option value="torch">PyTorch .pt checkpoint</option>
          </select>
        </label>
        <Slider
          label="Max new tokens"
          max={500}
          min={20}
          onChange={(event) => setMaxTokens(Number(event.target.value))}
          value={maxTokens}
          valueLabel={String(maxTokens)}
        />
        <Slider
          label="Temperature"
          max={2}
          min={0.1}
          onChange={(event) => setTemperature(Number(event.target.value))}
          step={0.1}
          value={temperature}
          valueLabel={temperature.toFixed(1)}
        />
        <Input label="Model label" onChange={(event) => setModelLabel(event.target.value)} value={modelLabel} />
      </div>
    </motion.aside>
  );
}
