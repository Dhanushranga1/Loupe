import { NavLink, Outlet } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true },
  { to: "/graph", label: "Symbol Graph" },
  { to: "/telemetry", label: "Telemetry" },
  { to: "/conventions", label: "Conventions" },
  { to: "/feedback", label: "Feedback" },
];

export function Layout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-text">
      <aside className="flex w-56 flex-shrink-0 flex-col border-r border-border bg-surface">
        <div className="flex items-center gap-2.5 px-5 py-4 border-b border-border">
          <span className="flex h-6 w-6 items-center justify-center rounded-full border-[1.5px] border-accent text-[13px] font-bold text-accent">
            L
          </span>
          <div>
            <div className="text-[13px] font-bold tracking-wide leading-none">Lens</div>
            <div className="text-[10px] text-text-faint font-mono leading-none mt-1">Loupe dashboard</div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 p-2">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-[13px] font-medium transition-colors ${
                  isActive
                    ? "bg-accent-soft text-accent"
                    : "text-text-dim hover:bg-surface-2 hover:text-text"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto p-3 text-[10px] text-text-faint font-mono leading-relaxed border-t border-border">
          Reads Loupe's own local index &amp; telemetry. No model calls, no API key.
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto scroll-thin">
        <Outlet />
      </main>
    </div>
  );
}
