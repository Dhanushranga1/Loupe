export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex h-40 items-center justify-center">
      <div className="flex flex-col items-center gap-2.5">
        <div className="scan-bar h-[3px] w-32 rounded-full bg-border" />
        <span className="font-mono text-[11px] uppercase tracking-wide text-text-faint">{label}</span>
      </div>
    </div>
  );
}

export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="panel-bracket rounded-[var(--radius-panel)] border border-bad/30 bg-bad-soft px-4 py-3 text-sm">
      <div className="font-mono text-[11px] uppercase tracking-wide text-bad font-semibold mb-1">
        Couldn't reach the Loupe server
      </div>
      <div className="text-text-dim">{message}</div>
      <div className="mt-2 font-mono text-[11px] text-text-faint">
        Is <code className="mono">loupe serve</code> running? Lens expects it at{" "}
        <code className="mono">http://127.0.0.1:8765</code> by default.
      </div>
    </div>
  );
}

export function Empty({ label }: { label: string }) {
  return (
    <div className="flex h-40 flex-col items-center justify-center gap-2 text-text-faint">
      <span className="h-6 w-6 rounded-full border border-dashed border-text-faint/40" aria-hidden />
      <span className="font-mono text-[11px]">{label}</span>
    </div>
  );
}
