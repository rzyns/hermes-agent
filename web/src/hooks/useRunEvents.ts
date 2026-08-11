import { useCallback, useEffect, useRef, useState } from "react";
import { HERMES_BASE_PATH } from "@/lib/api";

/**
 * useRunEvents — live SSE consumer for the dashboard run-events stream.
 *
 * Connects to the dashboard's SSE proxy (``/api/runs/{run_id}/events``),
 * which forwards to the api_server's ``GET /v1/runs/{run_id}/events``.
 *
 * Contract gotcha (recorded by the first producer reviewer): all replayable
 * frames are NAMED event frames, so ``EventSource.onmessage`` will NEVER
 * fire. We register ``addEventListener`` per event name from the contract
 * event map, including the terminal frame and control frames. Native
 * EventSource reconnect sends ``Last-Event-ID`` automatically and only
 * replayable frames carry ``id``, which matches the contract — so we let
 * the browser handle reconnect rather than hand-rolling it.
 *
 * Degrade gracefully when the endpoint is absent or errors: the hook exposes
 * ``status`` and ``error`` so the UI can show a non-breaking message.
 */

// ── Contract event names (§4 category map) ──────────────────────────
// Replayable run events (carry per-run monotonic ``id``):
export const RUN_EVENT_NAMES = [
  "tool.started",
  "tool.completed",
  "reasoning.available",
  "subagent.start",
  "subagent.complete",
  "message.delta",
  "approval.request",
  "approval.resolved",
  "run.queued",
  "run.running",
  "run.completed",
  "run.failed",
  "run.cancelled",
  "error",
  "hermes.run_events.event.oversize",
] as const;

// Control/meta frames (mandatory, never filtered, no ``id``):
const CONTROL_EVENT_NAMES = [
  "hermes.run_events.capabilities",
  "hermes.run_events.heartbeat",
  "hermes.run_events.subscriber.lagged",
  "hermes.run_events.terminal",
  // Proxy-originated synthetic frame (dashboard → browser):
  "hermes.run_events.proxy_error",
] as const;

const ALL_EVENT_NAMES = [...RUN_EVENT_NAMES, ...CONTROL_EVENT_NAMES] as const;

export interface RunEvent {
  /** Event name from the SSE ``event:`` field. */
  name: string;
  /** Parsed JSON payload from the SSE ``data:`` field. */
  data: Record<string, unknown>;
  /** Per-run monotonic ID (only present on replayable events). */
  id?: number;
  /** Wall-clock receipt timestamp (ms since epoch). */
  receivedAt: number;
}

export interface RunEventsCapabilities {
  version?: number;
  snapshot_id?: string;
  max_event_bytes?: number;
  max_replay_events?: number;
  max_replay_bytes?: number;
  max_subscriber_queue_events?: number;
  max_subscriber_queue_bytes?: number;
  max_concurrent_subscribers?: number;
  max_subscribers_per_run?: number;
  max_retained_runs?: number;
  max_active_runs_for_events?: number;
  heartbeat_seconds?: number;
  terminal_retention_seconds?: number;
  supported_include_categories?: string[];
  mandatory_category?: string;
  include_match?: string;
  [key: string]: unknown;
}

export type RunEventsStatus =
  | "idle"
  | "connecting"
  | "open"
  | "lagged"
  | "terminated"
  | "error";

export interface UseRunEventsResult {
  events: RunEvent[];
  capabilities: RunEventsCapabilities | null;
  status: RunEventsStatus;
  error: string | null;
  lastEventId: number | null;
  /** Total events received (excludes heartbeats). */
  eventCount: number;
  /** Clear the accumulated event buffer. */
  clearEvents: () => void;
}

export interface UseRunEventsOptions {
  /** Run ID to subscribe to. When empty/null, the hook stays idle. */
  runId: string | null | undefined;
  /** Category include filter (comma-separated, e.g. "tool,message,status"). */
  include?: string;
  /** Max events to retain in the buffer (default 500). */
  maxBuffer?: number;
  /** Whether to auto-connect (default true). */
  enabled?: boolean;
}

/**
 * Build the SSE proxy URL with auth + query params.
 * EventSource cannot set custom headers, so loopback mode authenticates via
 * ``?token=<session_token>``. The ``include`` category filter is forwarded
 * as a query param to the upstream api_server.
 */
function buildSseUrl(runId: string, include?: string): string {
  const base = HERMES_BASE_PATH;
  const params = new URLSearchParams();
  // Loopback mode: session token as query param (gated mode uses cookies).
  // The global is declared in api.ts and re-exported via the import above.
  const token =
    typeof window !== "undefined"
      ? (window as Window & typeof globalThis & {
          __HERMES_SESSION_TOKEN__?: string;
        }).__HERMES_SESSION_TOKEN__
      : undefined;
  if (token) params.set("token", token);
  if (include) params.set("include", include);
  const qs = params.toString();
  const encodedRunId = encodeURIComponent(runId);
  return `${base}/api/runs/${encodedRunId}/events${qs ? `?${qs}` : ""}`;
}

export function useRunEvents(
  options: UseRunEventsOptions,
): UseRunEventsResult {
  const { runId, include, maxBuffer = 500, enabled = true } = options;
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [capabilities, setCapabilities] =
    useState<RunEventsCapabilities | null>(null);
  const [status, setStatus] = useState<RunEventsStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastEventId, setLastEventId] = useState<number | null>(null);
  const [eventCount, setEventCount] = useState(0);

  const sourceRef = useRef<EventSource | null>(null);
  // Track whether the stream has terminated so we don't auto-reconnect
  // after a terminal frame (the run is over).
  const terminatedRef = useRef(false);
  // Track lag so we don't overwrite a lagged status with onopen reconnect.
  const laggedRef = useRef(false);

  const clearEvents = useCallback(() => {
    setEvents([]);
    setEventCount(0);
    setLastEventId(null);
  }, []);

  useEffect(() => {
    // Reset state when runId changes.  The eslint disable mirrors the
    // established pattern in PageHeaderProvider: an effect that clears stale
    // per-subscription state on a key change IS the correct sync here.
    /* eslint-disable react-hooks/set-state-in-effect */
    terminatedRef.current = false;
    laggedRef.current = false;
    setStatus("idle");
    setError(null);
    setCapabilities(null);
    setEvents([]);
    setEventCount(0);
    setLastEventId(null);
    /* eslint-enable react-hooks/set-state-in-effect */

    if (!enabled || !runId) {
      return;
    }

    // Close any existing source before opening a new one.
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }

    const url = buildSseUrl(runId, include);
    setStatus("connecting");
    let es: EventSource;
    try {
      es = new EventSource(url, { withCredentials: true });
    } catch (err) {
      setStatus("error");
      setError(
        err instanceof Error ? err.message : "Failed to open EventSource",
      );
      return;
    }
    sourceRef.current = es;

    es.onopen = () => {
      if (!terminatedRef.current && !laggedRef.current) {
        setStatus("open");
        setError(null);
      }
    };

    es.onerror = () => {
      // EventSource auto-reconnects unless we close it. If the stream
      // terminated or lagged, the proxy already sent a synthetic close
      // frame; the browser will retry, but we keep the terminal/lagged
      // status until the user explicitly reconnects or the run resumes.
      if (terminatedRef.current) {
        es.close();
        setStatus("terminated");
        return;
      }
      if (laggedRef.current) {
        // Native reconnect will resume from Last-Event-ID; status stays
        // "lagged" until the reconnect succeeds (onopen clears it).
        return;
      }
      // Unprovoked error — likely the api_server is down. EventSource will
      // retry, but surface the degraded state.
      setStatus("error");
      setError("Connection lost — retrying automatically");
    };

    // ── Register named listeners per the contract event map ──────────
    // ALL frames are named (event: <name>), so onmessage never fires.
    const handler = (name: string) => (ev: MessageEvent) => {
      let payload: Record<string, unknown> = {};
      try {
        payload = ev.data ? JSON.parse(ev.data) : {};
      } catch {
        payload = { _raw: ev.data };
      }

      // Capabilities control frame — parse and store, don't add to feed.
      if (name === "hermes.run_events.capabilities") {
        setCapabilities(payload as RunEventsCapabilities);
        return;
      }

      // Heartbeat — keepalive comment frame, ignore.
      if (name === "hermes.run_events.heartbeat") {
        return;
      }

      // Lagged — the subscriber was closed by the producer. Let native
      // EventSource reconnect from Last-Event-ID (the contract guarantees
      // only replayable frames carry id, so Last-Event-ID is correct).
      if (name === "hermes.run_events.subscriber.lagged") {
        laggedRef.current = true;
        setStatus("lagged");
        return;
      }

      // Terminal — the run ended. Close the stream and surface termination.
      if (name === "hermes.run_events.terminal") {
        terminatedRef.current = true;
        setStatus("terminated");
        es.close();
        return;
      }

      // Proxy error — the dashboard proxy could not reach the api_server
      // or the upstream returned an error. Surface it without crashing.
      if (name === "hermes.run_events.proxy_error") {
        const reason =
          (payload.reason as string) ||
          (payload.upstream_status != null
            ? `upstream status ${payload.upstream_status}`
            : "proxy error");
        setStatus("error");
        setError(reason);
        // The proxy closes the stream after this frame; EventSource will
        // retry. Mark as non-terminal so the user can reconnect.
        return;
      }

      // Replayable event — add to the feed.
      const id = ev.lastEventId ? parseInt(ev.lastEventId, 10) : undefined;
      const event: RunEvent = {
        name,
        data: payload,
        id: Number.isFinite(id) ? id : undefined,
        receivedAt: Date.now(),
      };
      setEvents((prev) => {
        const next = [...prev, event];
        return next.length > maxBuffer
          ? next.slice(next.length - maxBuffer)
          : next;
      });
      setEventCount((c) => c + 1);
      if (Number.isFinite(id)) {
        setLastEventId(id as number);
      }
    };

    for (const name of ALL_EVENT_NAMES) {
      es.addEventListener(name, handler(name) as EventListener);
    }

    return () => {
      es.close();
      if (sourceRef.current === es) {
        sourceRef.current = null;
      }
    };
    // include is part of the URL; changing it reconnects.
  }, [runId, include, enabled, maxBuffer]);

  return {
    events,
    capabilities,
    status,
    error,
    lastEventId,
    eventCount,
    clearEvents,
  };
}
