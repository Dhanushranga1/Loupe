export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex h-40 items-center justify-center text-sm text-text-faint font-mono">
      <span className="inline-block h-3 w-3 animate-pulse rounded-full bg-accent mr-3" />
      {label}
    </div>
  );
}

export function ErrorPanel({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad font-mono">
      <div className="font-semibold mb-1">Couldn't reach the Loupe server</div>
      <div className="text-text-dim">{message}</div>
      <div className="mt-2 text-xs text-text-faint">
        Is <code className="mono">loupe serve</code> running? Lens expects it at{" "}
        <code className="mono">http://127.0.0.1:8765</code> by default.
      </div>
    </div>
  );
}

export function Empty({ label }: { label: string }) {
  return <div className="flex h-40 items-center justify-center text-sm text-text-faint">{label}</div>;
}
