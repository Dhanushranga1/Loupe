import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api } from "../api";

function Readout({ label, value, accent = false }: { label: string; value: string | number; accent?: boolean }) {
  return (
    <div className="group relative overflow-hidden rounded-[var(--radius-panel)] border border-border bg-surface px-5 py-4">
      <div className="absolute inset-x-0 top-0 h-[2px] bg-accent opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="text-[10px] font-mono uppercase tracking-wide text-text-faint">{label}</div>
      <div className={`mt-1.5 text-[26px] font-semibold tabular-nums font-mono leading-none ${accent ? "text-accent" : ""}`}>
        {value}
      </div>
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
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Readout label="Symbols" value={status.data.symbol_count} accent />
              <Readout label="Files" value={status.data.file_count} />
              <Readout label="Unresolved refs" value={status.data.unresolved_reference_count} />
              <Readout label="Languages" value={status.data.languages.join(", ") || "—"} />
            </div>
            <div className="panel-bracket rounded-[var(--radius-panel)] border border-border bg-surface px-5 py-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wide text-text-faint mb-1.5">Repo root</div>
                  <div className="font-mono text-[13px] break-all text-text-dim">{status.data.repo_root}</div>
                </div>
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-wide text-text-faint mb-1.5">Last indexed</div>
                  <div className="font-mono text-[13px] text-text-dim">
                    {status.data.last_indexed ? new Date(status.data.last_indexed).toLocaleString() : "never"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
