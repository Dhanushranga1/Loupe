import { useMemo, useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel, Empty } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, type TelemetryEntry } from "../api";

function relativeTime(unixSeconds: number): string {
  const diffMs = Date.now() - unixSeconds * 1000;
  const diffS = Math.round(diffMs / 1000);
  if (diffS < 60) return `${diffS}s ago`;
  if (diffS < 3600) return `${Math.round(diffS / 60)}m ago`;
  if (diffS < 86400) return `${Math.round(diffS / 3600)}h ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString();
}

function FeedbackButtons({ logId, existing }: { logId: string; existing?: "helpful" | "not_helpful" }) {
  const [submitted, setSubmitted] = useState<"helpful" | "not_helpful" | undefined>(existing);
  const [pending, setPending] = useState(false);

  async function submit(rating: "helpful" | "not_helpful") {
    setPending(true);
    try {
      await api.submitFeedback(logId, rating);
      setSubmitted(rating);
    } catch {
      // Best-effort: leave buttons active so the user can retry. The
      // dashboard has no toast system yet — a real gap, not silently hidden.
    } finally {
      setPending(false);
    }
  }

  if (submitted) {
    return (
      <span
        className={`rounded px-2 py-0.5 text-[11px] font-medium ${
          submitted === "helpful" ? "bg-good-soft text-good" : "bg-bad-soft text-bad"
        }`}
      >
        {submitted === "helpful" ? "Marked helpful" : "Marked not helpful"}
      </span>
    );
  }

  return (
    <div className="flex gap-1">
      <button
        disabled={pending}
        onClick={() => submit("helpful")}
        className="rounded border border-border px-2 py-0.5 text-[11px] text-text-dim hover:border-good hover:text-good disabled:opacity-40"
      >
        Helpful
      </button>
      <button
        disabled={pending}
        onClick={() => submit("not_helpful")}
        className="rounded border border-border px-2 py-0.5 text-[11px] text-text-dim hover:border-bad hover:text-bad disabled:opacity-40"
      >
        Not helpful
      </button>
    </div>
  );
}

export function Telemetry() {
  const telemetry = useApi(() => api.telemetry(200));
  const feedback = useApi(api.feedback);
  const [toolFilter, setToolFilter] = useState<string>("all");

  const feedbackByLogId = useMemo(() => {
    const map = new Map<string, "helpful" | "not_helpful">();
    if (feedback.status === "ready") feedback.data.forEach((f) => map.set(f.retrieval_log_id, f.rating));
    return map;
  }, [feedback]);

  const toolNames = useMemo(() => {
    if (telemetry.status !== "ready") return [];
    return [...new Set(telemetry.data.map((e) => e.tool_name))].sort();
  }, [telemetry]);

  const rows = useMemo(() => {
    if (telemetry.status !== "ready") return [];
    return toolFilter === "all" ? telemetry.data : telemetry.data.filter((e) => e.tool_name === toolFilter);
  }, [telemetry, toolFilter]);

  return (
    <div>
      <PageHeader
        title="Telemetry"
        subtitle="Recent MCP tool calls, straight from .loupe/logs/retrieval/."
        right={
          telemetry.status === "ready" ? (
            <div className="font-mono text-[11px] text-text-faint">
              <span className="text-accent font-semibold tabular-nums">{telemetry.data.length}</span> calls logged
            </div>
          ) : undefined
        }
      />
      <div className="p-8">
        {telemetry.status === "loading" && <Loading label="Reading telemetry logs…" />}
        {telemetry.status === "error" && <ErrorPanel message={telemetry.error} />}
        {telemetry.status === "ready" && (
          <>
            <div className="mb-4 flex items-center gap-2">
              <span className="font-mono text-[11px] uppercase tracking-wide text-text-faint">Tool</span>
              <select
                value={toolFilter}
                onChange={(e) => setToolFilter(e.target.value)}
                className="rounded-[var(--radius-chip)] border border-border bg-surface-2 px-2 py-1 font-mono text-[12px] outline-none focus:border-accent"
              >
                <option value="all">all ({telemetry.data.length})</option>
                {toolNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>

            {rows.length === 0 ? (
              <Empty label="No tool calls recorded yet — connect Claude Code and make a query." />
            ) : (
              <div className="overflow-x-auto rounded-[var(--radius-panel)] border border-border">
                <table className="w-full border-collapse text-[12px]">
                  <thead>
                    <tr className="border-b border-border bg-surface-2 text-left text-[10px] uppercase tracking-wide text-text-faint">
                      <th className="px-3 py-2 font-medium">When</th>
                      <th className="px-3 py-2 font-medium">Tool</th>
                      <th className="px-3 py-2 font-medium">Query / target</th>
                      <th className="px-3 py-2 font-medium text-right">Latency</th>
                      <th className="px-3 py-2 font-medium text-right">Size</th>
                      <th className="px-3 py-2 font-medium">Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((entry: TelemetryEntry) => (
                      <tr key={entry.log_id} className="border-b border-border last:border-0 hover:bg-surface-2/50">
                        <td className="whitespace-nowrap px-3 py-2 font-mono text-text-faint">{relativeTime(entry.timestamp)}</td>
                        <td className="whitespace-nowrap px-3 py-2 font-mono font-medium">{entry.tool_name}</td>
                        <td className="max-w-xs truncate px-3 py-2 font-mono text-text-dim">
                          {entry.query_text ?? <span className="italic text-text-faint">—</span>}
                          {entry.error_code && <span className="ml-2 rounded bg-bad-soft px-1.5 py-0.5 text-bad">{entry.error_code}</span>}
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono tabular-nums text-text-dim">
                          {entry.latency_ms.toFixed(1)}ms
                        </td>
                        <td className="whitespace-nowrap px-3 py-2 text-right font-mono tabular-nums text-text-dim">
                          {entry.output_size_bytes}B
                        </td>
                        <td className="whitespace-nowrap px-3 py-2">
                          <FeedbackButtons logId={entry.log_id} existing={feedbackByLogId.get(entry.log_id)} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
