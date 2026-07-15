import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel, Empty } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api } from "../api";

export function Feedback() {
  const feedback = useApi(api.feedback);

  const helpfulCount = feedback.status === "ready" ? feedback.data.filter((f) => f.rating === "helpful").length : 0;
  const totalCount = feedback.status === "ready" ? feedback.data.length : 0;

  return (
    <div>
      <PageHeader
        title="Feedback"
        subtitle="Every submission here always outweighs Phase 6's proxy signal for the same retrieval — never blended."
      />
      <div className="p-8">
        {feedback.status === "loading" && <Loading label="Loading feedback…" />}
        {feedback.status === "error" && <ErrorPanel message={feedback.error} />}
        {feedback.status === "ready" && (
          <>
            {totalCount > 0 && (
              <div className="relative mb-4 overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface px-5 py-4">
                <div className="absolute inset-x-0 top-0 h-[2px] bg-accent opacity-70" />
                <span className="font-mono text-2xl font-semibold tabular-nums text-accent">{helpfulCount}</span>
                <span className="text-text-dim"> / </span>
                <span className="font-mono text-2xl font-semibold tabular-nums text-text-dim">{totalCount}</span>
                <span className="ml-2 font-mono text-[11px] uppercase tracking-wide text-text-faint">marked helpful</span>
              </div>
            )}

            {totalCount === 0 ? (
              <Empty label="No feedback submitted yet — use the buttons on the Telemetry page." />
            ) : (
              <div className="flex flex-col gap-2">
                {feedback.data.map((entry) => (
                  <div
                    key={`${entry.retrieval_log_id}-${entry.submitted_at}`}
                    className="flex items-center gap-3 rounded-[var(--radius-panel)] border border-border bg-surface px-4 py-3"
                  >
                    <span
                      className={`rounded px-2 py-0.5 text-[11px] font-medium ${
                        entry.rating === "helpful" ? "bg-good-soft text-good" : "bg-bad-soft text-bad"
                      }`}
                    >
                      {entry.rating === "helpful" ? "Helpful" : "Not helpful"}
                    </span>
                    <span className="truncate font-mono text-[12px] text-text-dim">{entry.retrieval_log_id}</span>
                    {entry.note && <span className="truncate text-[12px] text-text-faint">— {entry.note}</span>}
                    <span className="ml-auto whitespace-nowrap text-[11px] text-text-faint">
                      {entry.source === "dashboard" ? "via Lens" : "via Claude"}
                    </span>
                    <span className="whitespace-nowrap font-mono text-[11px] text-text-faint">
                      {new Date(entry.submitted_at * 1000).toLocaleString()}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
