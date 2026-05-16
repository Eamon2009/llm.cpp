import { useHealth } from "../api/health";

export function useConnectionStatus() {
  const { data, isError, isLoading } = useHealth();
  return {
    isLoading,
    isOnline: Boolean(data?.cpp_server === "ok" && !isError),
    isApiOnline: Boolean(data?.api === "ok" && !isError),
  };
}
