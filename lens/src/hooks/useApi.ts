import { useEffect, useState, useCallback } from "react";

type State<T> =
  | { status: "loading"; data?: undefined; error?: undefined }
  | { status: "error"; data?: undefined; error: string }
  | { status: "ready"; data: T; error?: undefined };

/** Fetches `fn()` on mount and whenever `deps` change, with a `refetch` escape hatch
 * for post-mutation refresh (e.g. after submitting feedback). Cancels a stale
 * in-flight request's effect on unmount/deps-change so a slow first load can't
 * clobber a faster subsequent one. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []): State<T> & { refetch: () => void } {
  const [state, setState] = useState<State<T>>({ status: "loading" });
  const [nonce, setNonce] = useState(0);

  const load = useCallback(() => {
    let cancelled = false;
    setState({ status: "loading" });
    fn()
      .then((data) => {
        if (!cancelled) setState({ status: "ready", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) setState({ status: "error", error: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  useEffect(() => load(), [load]);

  return { ...state, refetch: () => setNonce((n) => n + 1) } as State<T> & { refetch: () => void };
}
