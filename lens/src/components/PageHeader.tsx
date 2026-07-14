export function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="border-b border-border px-8 py-5">
      <h1 className="text-lg font-bold tracking-tight">{title}</h1>
      {subtitle && <p className="mt-1 text-[13px] text-text-dim">{subtitle}</p>}
    </div>
  );
}
