import { useEffect, useState } from "react";
import { BASE_URL } from "../api";

export type ConnectionStatus = "checking" | "online" | "offline";

const POLL_INTERVAL_MS = 8000;

/** Pings the plain /loupe/version endpoint (not a dashboard-specific route — the
 * same one any client, including Claude Code's MCP handshake, can reach) on an
 * interval, so the sidebar's status dot reflects whether `loupe serve` is
 * actually up right now, not just whether the last page load happened to work. */
export function useConnectionStatus(): ConnectionStatus {
  const [status, setStatus] = useState<ConnectionStatus>("checking");

  useEffect(() => {
    let cancelled = false;

    async function ping() {
      try {
        const res = await fetch(`${BASE_URL}/loupe/version`, { signal: AbortSignal.timeout(3000) });
        if (!cancelled) setStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    }

    ping();
    const id = setInterval(ping, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return status;
}
