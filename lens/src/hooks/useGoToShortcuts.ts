import { useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

// GitHub/vim-style "g then <letter>" chords — a small, real developer-tool
// touch (most consumer dashboards don't have any keyboard navigation at all).
const ROUTES: Record<string, string> = {
  o: "/",
  g: "/graph",
  t: "/telemetry",
  c: "/conventions",
  f: "/feedback",
};
const CHORD_WINDOW_MS = 900;

export function useGoToShortcuts() {
  const navigate = useNavigate();
  const armed = useRef(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const typing = target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;

      if (armed.current) {
        armed.current = false;
        clearTimeout(timer.current);
        const path = ROUTES[e.key];
        if (path) {
          e.preventDefault();
          navigate(path);
        }
        return;
      }

      if (e.key === "g") {
        armed.current = true;
        timer.current = setTimeout(() => {
          armed.current = false;
        }, CHORD_WINDOW_MS);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      clearTimeout(timer.current);
    };
  }, [navigate]);
}
