import { formatDistanceToNow } from "date-fns";

export function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString();
}

export function formatRelativeTime(value: string): string {
  return formatDistanceToNow(new Date(value), { addSuffix: true }).replace("less than a minute ago", "just now");
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60);
  return `${minutes}m ${remaining}s`;
}
