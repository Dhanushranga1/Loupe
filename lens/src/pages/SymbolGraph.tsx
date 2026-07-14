import { useEffect, useMemo, useRef, useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { Loading, ErrorPanel } from "../components/StateViews";
import { useApi } from "../hooks/useApi";
import { api, type GraphEdge, type GraphNode } from "../api";
import { ForceGraph, type SimNode } from "../graph/ForceGraph";

const MODULE_VAR: Record<string, string> = {
  app: "--mod-app",
  eval: "--mod-eval",
  retrieval: "--mod-retrieval",
  parsing: "--mod-parsing",
  graph: "--mod-graph",
  governor: "--mod-governor",
  storage: "--mod-storage",
  bandit: "--mod-bandit",
};

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function colorForModule(mod: string): string {
  const varName = MODULE_VAR[mod];
  if (varName) return cssVar(varName);
  // deterministic fallback hue for modules outside Loupe's own 8, so an
  // arbitrary indexed repo still gets a stable (if generic) palette
  let hash = 0;
  for (let i = 0; i < mod.length; i++) hash = (hash * 31 + mod.charCodeAt(i)) >>> 0;
  return `hsl(${hash % 360}, 40%, 55%)`;
}

interface RelatedSymbol {
  id: string;
  name: string;
  module: string;
}

function GraphInner({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const graphRef = useRef<ForceGraph | null>(null);
  const [selected, setSelected] = useState<SimNode | null>(null);
  const [search, setSearch] = useState("");
  const [activeModules, setActiveModules] = useState<Set<string>>(() => new Set(nodes.map((n) => n.module)));

  const modules = useMemo(() => [...new Set(nodes.map((n) => n.module))].sort(), [nodes]);
  const moduleCounts = useMemo(() => {
    const counts = new Map<string, number>();
    nodes.forEach((n) => counts.set(n.module, (counts.get(n.module) ?? 0) + 1));
    return counts;
  }, [nodes]);

  const nodesById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const outgoing = useMemo(() => {
    const map = new Map<string, RelatedSymbol[]>();
    edges.forEach((e) => {
      const target = nodesById.get(e.target);
      if (!target) return;
      const list = map.get(e.source) ?? [];
      list.push({ id: target.id, name: target.name, module: target.module });
      map.set(e.source, list);
    });
    return map;
  }, [edges, nodesById]);
  const incoming = useMemo(() => {
    const map = new Map<string, RelatedSymbol[]>();
    edges.forEach((e) => {
      const source = nodesById.get(e.source);
      if (!source) return;
      const list = map.get(e.target) ?? [];
      list.push({ id: source.id, name: source.name, module: source.module });
      map.set(e.target, list);
    });
    return map;
  }, [edges, nodesById]);

  useEffect(() => {
    if (!canvasRef.current) return;
    const graph = new ForceGraph(canvasRef.current, nodes, edges, {
      onSelect: setSelected,
      onHover: () => {},
      colorFor: colorForModule,
      cssVar,
    });
    graphRef.current = graph;
    return () => graph.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges]);

  useEffect(() => {
    graphRef.current?.setSearch(search);
  }, [search]);

  useEffect(() => {
    graphRef.current?.setModuleFilter(activeModules);
  }, [activeModules]);

  function toggleModule(mod: string) {
    setActiveModules((prev) => {
      const next = new Set(prev);
      if (next.has(mod)) next.delete(mod);
      else next.add(mod);
      return next;
    });
  }

  function focusOn(id: string) {
    graphRef.current?.focusOn(id);
  }

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} className="block h-full w-full cursor-grab active:cursor-grabbing" />

      <div className="absolute left-4 top-4 flex items-center gap-2">
        <div className="relative">
          <svg
            className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-faint"
            viewBox="0 0 16 16"
            fill="none"
          >
            <circle cx="7" cy="7" r="5.2" stroke="currentColor" strokeWidth="1.4" />
            <path d="M11 11L14.5 14.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Find a symbol…"
            className="w-60 rounded-md border border-border bg-surface-2 py-2 pl-8 pr-3 font-mono text-[13px] outline-none focus:border-accent"
          />
        </div>
      </div>

      <div className="absolute bottom-4 left-4 flex max-w-[200px] flex-col gap-1.5 rounded-xl border border-border bg-surface p-3 shadow-lg">
        <div className="mb-0.5 text-[10px] uppercase tracking-wide text-text-faint">Modules</div>
        {modules.map((mod) => (
          <button
            key={mod}
            onClick={() => toggleModule(mod)}
            className={`flex items-center gap-2 rounded px-1 py-0.5 text-left font-mono text-[12px] ${
              activeModules.has(mod) ? "text-text-dim hover:bg-surface-2" : "opacity-35 hover:bg-surface-2"
            }`}
          >
            <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ background: colorForModule(mod) }} />
            <span>{mod}</span>
            <span className="ml-auto text-[11px] text-text-faint">{moduleCounts.get(mod)}</span>
          </button>
        ))}
      </div>

      <div className="absolute bottom-4 right-4 rounded-lg border border-border bg-surface px-3 py-2 font-mono text-[11px] leading-relaxed text-text-faint">
        drag canvas to pan · scroll to zoom
        <br />
        drag a node to move it · click to inspect
      </div>

      {selected && (
        <div className="absolute right-4 top-4 flex max-h-[calc(100%-2rem)] w-80 flex-col overflow-hidden rounded-xl border border-border bg-surface shadow-xl">
          <div className="relative border-b border-border p-4">
            <button
              onClick={() => graphRef.current?.select(null)}
              className="absolute right-3 top-3 text-lg leading-none text-text-faint hover:text-text"
            >
              &times;
            </button>
            <div className="mb-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-text-faint">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: colorForModule(selected.module) }} />
              {selected.module} · {selected.kind}
            </div>
            <div className="break-words font-mono text-[15px] font-semibold">{selected.name}</div>
            <div className="mt-1.5 break-all font-mono text-[11px] text-text-dim">
              {selected.file_path}:{selected.line_start}
            </div>
          </div>
          <div className="scroll-thin overflow-y-auto p-4">
            <div className="flex gap-3.5 mb-3">
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-wide text-text-faint">Pagerank</div>
                <div className="mt-0.5 font-mono text-base font-semibold tabular-nums">{selected.pagerank.toFixed(4)}</div>
              </div>
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-wide text-text-faint">Lines</div>
                <div className="mt-0.5 font-mono text-base font-semibold tabular-nums">
                  {selected.line_start}–{selected.line_end}
                </div>
              </div>
            </div>
            <RelList title="Calls" items={outgoing.get(selected.id) ?? []} onPick={focusOn} />
            <RelList title="Called by" items={incoming.get(selected.id) ?? []} onPick={focusOn} />
          </div>
        </div>
      )}
    </div>
  );
}

function RelList({ title, items, onPick }: { title: string; items: RelatedSymbol[]; onPick: (id: string) => void }) {
  return (
    <div className="mb-3">
      <div className="mb-1.5 flex justify-between text-[10px] uppercase tracking-wide text-text-faint">
        <span>{title}</span>
        <span>{items.length}</span>
      </div>
      {items.length === 0 ? (
        <div className="py-1 text-[12px] italic text-text-faint">none in this graph</div>
      ) : (
        <ul className="flex flex-col gap-0.5">
          {items.map((item) => (
            <li key={item.id}>
              <button
                onClick={() => onPick(item.id)}
                className="flex w-full items-center gap-1.5 truncate rounded px-1.5 py-1 text-left font-mono text-[12px] text-text-dim hover:bg-surface-2 hover:text-text"
              >
                <span className="h-1.5 w-1.5 flex-shrink-0 rounded-full" style={{ background: colorForModule(item.module) }} />
                <span className="truncate">{item.name}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function SymbolGraph() {
  const graph = useApi(api.graph);

  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Symbol Graph" subtitle="Live call graph of the served repo — drag, zoom, click to inspect." />
      <div className="relative flex-1">
        {graph.status === "loading" && <Loading label="Loading symbol graph…" />}
        {graph.status === "error" && (
          <div className="p-8">
            <ErrorPanel message={graph.error} />
          </div>
        )}
        {graph.status === "ready" && <GraphInner nodes={graph.data.nodes} edges={graph.data.edges} />}
      </div>
    </div>
  );
}
