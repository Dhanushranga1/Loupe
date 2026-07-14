// A small, dependency-free force-directed graph renderer over <canvas>.
// Framework-agnostic on purpose: React only owns mount/unmount (see
// pages/SymbolGraph.tsx's useEffect) and passes callbacks for
// selection/hover; this class owns the simulation, camera, and picking.

import type { GraphEdge, GraphNode } from "../api";

export interface SimNode extends GraphNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  visible: boolean;
  pinned: boolean;
}

interface SimEdge {
  source: SimNode;
  target: SimNode;
  type: string;
}

export interface ForceGraphCallbacks {
  onSelect: (node: SimNode | null) => void;
  onHover: (node: SimNode | null) => void;
  colorFor: (moduleName: string) => string;
  cssVar: (name: string) => string;
}

const ALPHA_DECAY = 0.0062;
const ALPHA_MIN = 0.006;

export class ForceGraph {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private nodes: SimNode[] = [];
  private edges: SimEdge[] = [];
  private nodesById = new Map<string, SimNode>();
  private adjacency = new Map<string, { out: string[]; in: string[] }>();
  private cam = { x: 0, y: 0, scale: 0.62 };
  private dpr = Math.min(window.devicePixelRatio || 1, 2);
  private alpha = 1;
  private selected: SimNode | null = null;
  private hovered: SimNode | null = null;
  private searchMatches = new Set<string>();
  private dragging: SimNode | null = null;
  private panning = false;
  private panStart: [number, number] | null = null;
  private camStart = { x: 0, y: 0 };
  private downPos: [number, number] | null = null;
  private movedSinceDown = false;
  private raf = 0;
  private cb: ForceGraphCallbacks;
  private resizeObserver: ResizeObserver;

  constructor(canvas: HTMLCanvasElement, nodes: GraphNode[], edges: GraphEdge[], cb: ForceGraphCallbacks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.cb = cb;
    this.setData(nodes, edges);
    this.resize();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement!);

    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerup", this.onPointerUp);
    canvas.addEventListener("pointercancel", this.onPointerUp);
    canvas.addEventListener("wheel", this.onWheel, { passive: false });

    this.loop();
  }

  setData(nodes: GraphNode[], edges: GraphEdge[]) {
    this.nodesById.clear();
    const modules = [...new Set(nodes.map((n) => n.module))];
    const modAngle = new Map(modules.map((m, i) => [m, (i / modules.length) * Math.PI * 2]));
    const R = 520;

    this.nodes = nodes.map((n) => {
      const base = modAngle.get(n.module) ?? 0;
      const a = base + (Math.random() - 0.5) * 1.1;
      const r = R * (0.35 + Math.random() * 0.65);
      const node: SimNode = {
        ...n,
        x: Math.cos(a) * r,
        y: Math.sin(a) * r,
        vx: 0,
        vy: 0,
        r: 4 + Math.sqrt(Math.max(n.pagerank, 0.0001)) * 220,
        visible: true,
        pinned: false,
      };
      this.nodesById.set(n.id, node);
      return node;
    });

    this.edges = edges
      .map((e) => ({ source: this.nodesById.get(e.source), target: this.nodesById.get(e.target), type: e.type }))
      .filter((e): e is SimEdge => !!e.source && !!e.target);

    this.adjacency.clear();
    this.nodes.forEach((n) => this.adjacency.set(n.id, { out: [], in: [] }));
    this.edges.forEach((e) => {
      this.adjacency.get(e.source.id)!.out.push(e.target.id);
      this.adjacency.get(e.target.id)!.in.push(e.source.id);
    });

    this.alpha = 1;
  }

  setModuleFilter(active: Set<string>) {
    this.nodes.forEach((n) => (n.visible = active.has(n.module)));
    this.alpha = Math.max(this.alpha, 0.3);
  }

  setSearch(term: string) {
    if (!term) {
      this.searchMatches = new Set();
      return;
    }
    const lower = term.toLowerCase();
    this.searchMatches = new Set(
      this.nodes.filter((n) => n.visible && n.name.toLowerCase().includes(lower)).map((n) => n.id)
    );
  }

  select(id: string | null) {
    this.selected = id ? this.nodesById.get(id) ?? null : null;
    this.cb.onSelect(this.selected);
  }

  focusOn(id: string) {
    const node = this.nodesById.get(id);
    if (!node) return;
    this.cam.x = node.x;
    this.cam.y = node.y;
    this.alpha = Math.max(this.alpha, 0.15);
    this.select(id);
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("pointercancel", this.onPointerUp);
    this.canvas.removeEventListener("wheel", this.onWheel);
  }

  private resize = () => {
    const rect = this.canvas.parentElement!.getBoundingClientRect();
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.canvas.style.width = rect.width + "px";
    this.canvas.style.height = rect.height + "px";
  };

  private worldToScreen(x: number, y: number): [number, number] {
    const w = this.canvas.width,
      h = this.canvas.height;
    return [w / 2 + (x - this.cam.x) * this.cam.scale * this.dpr, h / 2 + (y - this.cam.y) * this.cam.scale * this.dpr];
  }

  private screenToWorld(sx: number, sy: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const w = this.canvas.width,
      h = this.canvas.height;
    const px = (sx - rect.left) * this.dpr,
      py = (sy - rect.top) * this.dpr;
    return [this.cam.x + (px - w / 2) / (this.cam.scale * this.dpr), this.cam.y + (py - h / 2) / (this.cam.scale * this.dpr)];
  }

  private tick() {
    if (this.alpha <= ALPHA_MIN) return;
    const n = this.nodes.length;
    for (let i = 0; i < n; i++) {
      const a = this.nodes[i];
      if (!a.visible) continue;
      for (let j = i + 1; j < n; j++) {
        const b = this.nodes[j];
        if (!b.visible) continue;
        let dx = a.x - b.x,
          dy = a.y - b.y;
        let dist2 = dx * dx + dy * dy;
        if (dist2 < 0.01) {
          dx = Math.random() - 0.5;
          dy = Math.random() - 0.5;
          dist2 = 0.01;
        }
        const dist = Math.sqrt(dist2);
        const force = (2600 / dist2) * this.alpha;
        const fx = (dx / dist) * force,
          fy = (dy / dist) * force;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
    }
    this.edges.forEach((e) => {
      if (!e.source.visible || !e.target.visible) return;
      let dx = e.target.x - e.source.x,
        dy = e.target.y - e.source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const force = ((dist - 70) * 0.02) * this.alpha;
      const fx = (dx / dist) * force,
        fy = (dy / dist) * force;
      e.source.vx += fx;
      e.source.vy += fy;
      e.target.vx -= fx;
      e.target.vy -= fy;
    });
    this.nodes.forEach((a) => {
      if (!a.visible) return;
      a.vx -= a.x * 0.0018 * this.alpha;
      a.vy -= a.y * 0.0018 * this.alpha;
    });
    this.nodes.forEach((a) => {
      if (a.pinned) {
        a.vx = 0;
        a.vy = 0;
        return;
      }
      a.vx *= 0.82;
      a.vy *= 0.82;
      a.x += a.vx;
      a.y += a.vy;
    });
    this.alpha -= ALPHA_DECAY;
  }

  private neighborsOf(node: SimNode | null): Set<string> | null {
    if (!node) return null;
    const adj = this.adjacency.get(node.id);
    if (!adj) return null;
    return new Set([...adj.out, ...adj.in, node.id]);
  }

  private draw() {
    const ctx = this.ctx;
    const w = this.canvas.width,
      h = this.canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = this.cb.cssVar("--bg");
    ctx.fillRect(0, 0, w, h);

    const focus = this.neighborsOf(this.selected) ?? this.neighborsOf(this.hovered);
    const searching = this.searchMatches.size > 0;

    ctx.lineWidth = Math.max(0.6, this.cam.scale * this.dpr * 0.9);
    this.edges.forEach((e) => {
      if (!e.source.visible || !e.target.visible) return;
      const inFocus = !focus || (focus.has(e.source.id) && focus.has(e.target.id));
      const [sx, sy] = this.worldToScreen(e.source.x, e.source.y);
      const [tx, ty] = this.worldToScreen(e.target.x, e.target.y);
      ctx.globalAlpha = inFocus ? (focus ? 0.9 : 0.5) : 0.5;
      ctx.strokeStyle = inFocus ? (focus ? this.cb.cssVar("--accent") : this.cb.cssVar("--border")) : this.cb.cssVar("--border");
      if (!inFocus && focus) return;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    this.nodes.forEach((n) => {
      if (!n.visible) return;
      const inFocus = !focus || focus.has(n.id);
      const isMatch = !searching || this.searchMatches.has(n.id);
      const [sx, sy] = this.worldToScreen(n.x, n.y);
      const r = Math.max(2, n.r * this.cam.scale * this.dpr);
      if (sx < -50 || sx > w + 50 || sy < -50 || sy > h + 50) return;

      let alphaMul = 1;
      if (focus && !inFocus) alphaMul = 0.12;
      if (searching && !isMatch) alphaMul = Math.min(alphaMul, 0.1);

      ctx.globalAlpha = alphaMul;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = this.cb.colorFor(n.module);
      ctx.fill();

      if (n === this.selected || (searching && isMatch)) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = (n === this.selected ? 2.4 : 1.6) * this.dpr;
        ctx.strokeStyle = this.cb.cssVar("--accent");
        ctx.beginPath();
        ctx.arc(sx, sy, r + (n === this.selected ? 3 : 2) * this.dpr, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      const showLabel = (r > 5.5 || n === this.selected || n === this.hovered || (searching && isMatch)) && alphaMul > 0.5;
      if (showLabel) {
        ctx.font = `${Math.max(10, 11 * this.dpr)}px ui-monospace, monospace`;
        ctx.fillStyle = n === this.selected ? this.cb.cssVar("--accent") : this.cb.cssVar("--text-dim");
        ctx.textBaseline = "middle";
        ctx.fillText(n.name.split(".").pop()!, sx + r + 5 * this.dpr, sy);
      }
    });
  }

  private loop = () => {
    this.tick();
    this.draw();
    this.raf = requestAnimationFrame(this.loop);
  };

  private nodeAtScreen(sx: number, sy: number): SimNode | null {
    const [wx, wy] = this.screenToWorld(sx, sy);
    let best: SimNode | null = null,
      bestDist = Infinity;
    for (const n of this.nodes) {
      if (!n.visible) continue;
      const dx = n.x - wx,
        dy = n.y - wy;
      const d2 = dx * dx + dy * dy;
      const rr = n.r + 4 / this.cam.scale;
      if (d2 < rr * rr && d2 < bestDist) {
        best = n;
        bestDist = d2;
      }
    }
    return best;
  }

  private onPointerDown = (ev: PointerEvent) => {
    this.canvas.setPointerCapture(ev.pointerId);
    this.downPos = [ev.clientX, ev.clientY];
    this.movedSinceDown = false;
    const hit = this.nodeAtScreen(ev.clientX, ev.clientY);
    if (hit) {
      this.dragging = hit;
      hit.pinned = true;
    } else {
      this.panning = true;
      this.panStart = [ev.clientX, ev.clientY];
      this.camStart = { x: this.cam.x, y: this.cam.y };
    }
  };

  private onPointerMove = (ev: PointerEvent) => {
    if (this.downPos) {
      const dx = ev.clientX - this.downPos[0],
        dy = ev.clientY - this.downPos[1];
      if (Math.abs(dx) + Math.abs(dy) > 3) this.movedSinceDown = true;
    }
    if (this.dragging) {
      const [wx, wy] = this.screenToWorld(ev.clientX, ev.clientY);
      this.dragging.x = wx;
      this.dragging.y = wy;
      this.alpha = Math.max(this.alpha, 0.25);
    } else if (this.panning && this.panStart) {
      const dx = ((ev.clientX - this.panStart[0]) * this.dpr) / (this.cam.scale * this.dpr);
      const dy = ((ev.clientY - this.panStart[1]) * this.dpr) / (this.cam.scale * this.dpr);
      this.cam.x = this.camStart.x - dx;
      this.cam.y = this.camStart.y - dy;
    } else {
      const hit = this.nodeAtScreen(ev.clientX, ev.clientY);
      if (hit !== this.hovered) {
        this.hovered = hit;
        this.cb.onHover(hit);
      }
    }
  };

  private onPointerUp = (ev: PointerEvent) => {
    if (this.dragging) {
      this.dragging.pinned = false;
      this.dragging = null;
    }
    this.panning = false;
    if (!this.movedSinceDown) {
      const hit = this.nodeAtScreen(ev.clientX, ev.clientY);
      this.select(hit ? hit.id : null);
    }
    this.downPos = null;
  };

  private onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    const factor = Math.exp(-ev.deltaY * 0.0012);
    const [wx, wy] = this.screenToWorld(ev.clientX, ev.clientY);
    this.cam.scale = Math.min(6, Math.max(0.08, this.cam.scale * factor));
    const [wx2, wy2] = this.screenToWorld(ev.clientX, ev.clientY);
    this.cam.x += wx - wx2;
    this.cam.y += wy - wy2;
  };
}
