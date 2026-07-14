import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api } from "../api";

function StatTile({ label, value, mono = true }: { label: string; value: string | number; mono?: boolean }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-5 py-4">
      <div className="text-[11px] uppercase tracking-wide text-text-faint">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  );
}

export function Overview() {
  const status = useApi(api.status);

  return (
    <div>
      <PageHeader title="Overview" subtitle="Live state of the currently-served repo index." />
      <div className="p-8">
        {status.status === "loading" && <Loading label="Reading index status…" />}
        {status.status === "error" && <ErrorPanel message={status.error} />}
        {status.status === "ready" && (
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
              <StatTile label="Symbols" value={status.data.symbol_count} />
              <StatTile label="Files" value={status.data.file_count} />
              <StatTile label="Unresolved refs" value={status.data.unresolved_reference_count} />
              <StatTile label="Languages" value={status.data.languages.join(", ") || "—"} />
            </div>
            <div className="rounded-xl border border-border bg-surface px-5 py-4">
              <div className="text-[11px] uppercase tracking-wide text-text-faint mb-2">Repo</div>
              <div className="font-mono text-sm break-all">{status.data.repo_root}</div>
              <div className="mt-3 text-[11px] uppercase tracking-wide text-text-faint mb-1">Last indexed</div>
              <div className="font-mono text-sm text-text-dim">
                {status.data.last_indexed ? new Date(status.data.last_indexed).toLocaleString() : "never"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
