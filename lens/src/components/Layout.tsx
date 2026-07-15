import { NavLink, Outlet } from "react-router-dom";
import { LoupeMark } from "./LoupeMark";
import { useConnectionStatus } from "../hooks/useConnectionStatus";
import { useGoToShortcuts } from "../hooks/useGoToShortcuts";

const NAV_ITEMS = [
  { to: "/", label: "Overview", end: true, chord: "o" },
  { to: "/graph", label: "Symbol graph", end: false, chord: "g" },
  { to: "/telemetry", label: "Telemetry", end: false, chord: "t" },
  { to: "/conventions", label: "Conventions", end: false, chord: "c" },
  { to: "/feedback", label: "Feedback", end: false, chord: "f" },
];

const STATUS_LABEL: Record<string, string> = {
  checking: "checking…",
  online: "connected",
  offline: "unreachable",
};

function StatusDot() {
  const status = useConnectionStatus();
  const color = status === "online" ? "bg-good" : status === "offline" ? "bg-bad" : "bg-text-faint";
  return (
    <div className="flex items-center gap-2 px-3 py-2.5 border-t border-border">
      <span className={`relative h-1.5 w-1.5 rounded-full ${color} ${status === "online" ? "status-live" : ""}`} />
      <span className="font-mono text-[10px] uppercase tracking-wide text-text-faint">{STATUS_LABEL[status]}</span>
    </div>
  );
}

export function Layout() {
  useGoToShortcuts();

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg text-text">
      <aside className="flex w-56 flex-shrink-0 flex-col border-r border-border bg-surface">
        <div className="flex items-center gap-2.5 px-4 py-4 border-b border-border">
          <LoupeMark size={20} className="text-accent flex-shrink-0" />
          <div>
            <div className="text-[13px] font-bold tracking-tight leading-none">Lens</div>
            <div className="text-[10px] text-text-faint font-mono leading-none mt-1">loupe dashboard</div>
          </div>
        </div>

        <nav className="flex flex-col gap-0.5 p-2 pt-3">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                `group relative flex items-center justify-between rounded-[3px] pl-3 pr-2 py-2 text-[13px] font-medium transition-colors border-l-2 ${
                  isActive
                    ? "border-accent bg-accent-soft text-accent"
                    : "border-transparent text-text-dim hover:bg-surface-2 hover:text-text"
                }`
              }
            >
              {item.label}
              <span className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <kbd className="key">g</kbd>
                <kbd className="key">{item.chord}</kbd>
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto">
          <StatusDot />
          <div className="p-3 pt-2 text-[10px] text-text-faint font-mono leading-relaxed border-t border-border">
            Reads Loupe's own local index &amp; telemetry. No model calls, no API key.
          </div>
        </div>
      </aside>
      <main className="relative flex-1 overflow-y-auto scroll-thin">
        <div className="grid-texture pointer-events-none fixed inset-y-0 right-0 left-56 -z-10" aria-hidden />
        <Outlet />
      </main>
    </div>
  );
}
