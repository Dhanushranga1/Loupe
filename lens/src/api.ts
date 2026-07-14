// Typed client for Loupe's plain REST /dashboard/* endpoints (server/app/dashboard.py).
// Deliberately not MCP: this app never speaks MCP protocol, just plain fetch()
// against the same FastAPI process Claude Code's MCP client also connects to.

const BASE_URL = import.meta.env.VITE_LOUPE_SERVER_URL ?? "http://127.0.0.1:8765";

export interface DashboardStatus {
  repo_root: string;
  symbol_count: number;
  file_count: number;
  unresolved_reference_count: number;
  languages: string[];
  last_indexed: string | null;
}

export interface GraphNode {
  id: string;
  name: string;
  kind: string;
  file_path: string;
  module: string;
  line_start: number;
  line_end: number;
  pagerank: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface DashboardGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ErrorHandlingConvention {
  majority_pattern: string | null;
  violation_count: number;
  violating_symbol_ids: string[];
}

export interface DocstringConvention {
  coverage_pct: number;
  dominant_style: string;
}

export interface ImportConvention {
  dominant_style: string;
  relative_count: number;
  absolute_count: number;
}

export interface ConventionsReport {
  error_handling: ErrorHandlingConvention;
  docstrings: DocstringConvention;
  imports: ImportConvention;
}

export interface TelemetryEntry {
  log_id: string;
  timestamp: number;
  session_id: string;
  turn_index: number;
  tool_name: string;
  query_text: string | null;
  query_intent: string | null;
  candidates: Array<{ symbol_id?: string; score?: number; value?: string }>;
  selected: Array<{ symbol_id?: string; score?: number; value?: string }>;
  latency_ms: number;
  output_size_bytes: number;
  error_code: string | null;
  outcome: { symbol_edited?: boolean } | null;
}

export interface FeedbackEntry {
  retrieval_log_id: string;
  rating: "helpful" | "not_helpful";
  note: string | null;
  submitted_at: number;
  source: "dashboard" | "claude_self_report";
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${path} -> HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  status: () => getJSON<DashboardStatus>("/dashboard/status"),
  graph: () => getJSON<DashboardGraph>("/dashboard/graph"),
  conventions: () => getJSON<ConventionsReport>("/dashboard/conventions"),
  telemetry: (limit = 200) => getJSON<TelemetryEntry[]>(`/dashboard/telemetry?limit=${limit}`),
  feedback: () => getJSON<FeedbackEntry[]>("/dashboard/feedback"),
  submitFeedback: async (retrieval_log_id: string, rating: "helpful" | "not_helpful", note?: string) => {
    const response = await fetch(`${BASE_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ retrieval_log_id, rating, note: note ?? null }),
    });
    if (!response.ok) {
      throw new Error(`submit feedback -> HTTP ${response.status}`);
    }
    return response.json() as Promise<{ status: string }>;
  },
};

export { BASE_URL };
