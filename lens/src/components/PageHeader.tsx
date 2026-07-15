import type { ReactNode } from "react";

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-6 border-b border-border px-8 py-5">
      <div className="flex items-start gap-3">
        <span className="mt-[3px] h-3.5 w-3.5 flex-shrink-0 border-l-2 border-t-2 border-accent" aria-hidden />
        <div>
          <h1 className="text-[15px] font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-1 text-[13px] text-text-dim max-w-xl">{subtitle}</p>}
        </div>
      </div>
      {right && <div className="flex-shrink-0">{right}</div>}
    </div>
  );
}
