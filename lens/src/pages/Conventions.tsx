import type { ReactNode } from "react";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api } from "../api";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5">
      <div className="mb-3 text-[11px] uppercase tracking-wide text-text-faint">{title}</div>
      {children}
    </div>
  );
}

export function Conventions() {
  const conventions = useApi(api.conventions);

  return (
    <div>
      <PageHeader
        title="Conventions"
        subtitle="Auto-derived, repo-wide — detection only, never enforced or auto-fixed."
      />
      <div className="p-8">
        {conventions.status === "loading" && <Loading label="Mining conventions…" />}
        {conventions.status === "error" && <ErrorPanel message={conventions.error} />}
        {conventions.status === "ready" && (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Section title="Error handling">
              {conventions.data.error_handling.majority_pattern ? (
                <>
                  <div className="mb-3 font-mono text-sm">{conventions.data.error_handling.majority_pattern}</div>
                  <div className="text-[12px] text-text-dim">
                    <span className="font-semibold text-text">{conventions.data.error_handling.violation_count}</span>{" "}
                    function{conventions.data.error_handling.violation_count === 1 ? "" : "s"} deviate
                    {conventions.data.error_handling.violation_count === 1 ? "s" : ""} from the majority pattern.
                  </div>
                  {conventions.data.error_handling.violating_symbol_ids.length > 0 && (
                    <div className="mt-2 max-h-32 overflow-y-auto scroll-thin rounded-md bg-surface-2 p-2 font-mono text-[11px] text-text-faint">
                      {conventions.data.error_handling.violating_symbol_ids.map((id) => (
                        <div key={id} className="truncate">
                          {id}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-[13px] italic text-text-faint">No try/except blocks found in this repo.</div>
              )}
            </Section>

            <Section title="Docstrings">
              <div className="mb-1 font-mono text-2xl font-semibold tabular-nums">
                {conventions.data.docstrings.coverage_pct.toFixed(0)}%
              </div>
              <div className="mb-3 text-[12px] text-text-dim">of public functions/classes documented</div>
              <div className="inline-flex items-center gap-1.5 rounded bg-accent-soft px-2 py-1 text-[12px] font-medium text-accent">
                {conventions.data.docstrings.dominant_style} style
              </div>
            </Section>

            <Section title="Imports">
              <div className="mb-2 inline-flex items-center gap-1.5 rounded bg-accent-soft px-2 py-1 text-[12px] font-medium text-accent">
                {conventions.data.imports.dominant_style}
              </div>
              <div className="flex gap-4 text-[12px] text-text-dim">
                <span>
                  <span className="font-mono font-semibold text-text">{conventions.data.imports.relative_count}</span> relative
                </span>
                <span>
                  <span className="font-mono font-semibold text-text">{conventions.data.imports.absolute_count}</span> absolute
                </span>
              </div>
            </Section>
          </div>
        )}
      </div>
    </div>
  );
}
