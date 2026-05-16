import { useEffect } from "react";

export function useKeyboardShortcut(keys: string[], callback: () => void): void {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      const key = event.key.toLowerCase();
      const matches = keys.every((item) => {
        const normalized = item.toLowerCase();
        if (normalized === "ctrl") {
          return event.ctrlKey;
        }
        if (normalized === "shift") {
          return event.shiftKey;
        }
        return key === normalized;
      });
      if (matches) {
        event.preventDefault();
        callback();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [callback, keys]);
}
