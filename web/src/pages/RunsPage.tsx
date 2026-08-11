import { type ComponentType, useCallback, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Pause,
  Play,
  Radio,
  Terminal,
  Trash2,
  XCircle,
  Zap,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { useRunEvents } from "@/hooks/useRunEvents";
import type { RunEvent, RunEventsStatus } from "@/hooks/useRunEvents";
import { cn } from "@/lib/utils";

// ── Category metadata (§4 category map) ────────────────────────────
interface CategoryMeta {
  label: string;
  color: string;
}

const CATEGORIES: Record<string, CategoryMeta> = {
  tool: { label: "Tool", color: "bg-blue-500/15 text-blue-400" },
  message: { label: "Message", color: "bg-cyan-500/15 text-cyan-400" },
  reasoning: { label: "Reasoning", color: "bg-purple-500/15 text-purple-400" },
  subagent: { label: "Subagent", color: "bg-indigo-500/15 text-indigo-400" },
  approval: { label: "Approval", color: "bg-amber-500/15 text-amber-400" },
  status: { label: "Status", color: "bg-green-500/15 text-green-400" },
  error: { label: "Error", color: "bg-red-500/15 text-red-400" },
};

const ALL_CATEGORIES = Object.keys(CATEGORIES);

// Map event name → category for display.
const EVENT_CATEGORY: Record<string, string> = {
  "tool.started": "tool",
  "tool.completed": "tool",
  "message.delta": "message",
  "reasoning.available": "reasoning",
  "subagent.start": "subagent",
  "subagent.complete": "subagent",
  "approval.request": "approval",
  "approval.resolved": "approval",
  "run.queued": "status",
  "run.running": "status",
  "run.completed": "status",
  "run.failed": "status",
  "run.cancelled": "status",
  error: "error",
  "hermes.run_events.event.oversize": "error",
};

function categoryForEvent(name: string): string {
  return EVENT_CATEGORY[name] ?? "status";
}

// ── Status badge ───────────────────────────────────────────────────
function StatusBadge({ status }: { status: RunEventsStatus }) {
  const map: Record<
    RunEventsStatus,
    { label: string; cls: string; icon: ComponentType<{ className?: string }> }
  > = {
    idle: { label: "Idle", cls: "bg-muted text-muted-foreground", icon: Pause },
    connecting: {
      label: "Connecting",
      cls: "bg-amber-500/15 text-amber-400",
      icon: Spinner,
    },
    open: {
      label: "Live",
      cls: "bg-green-500/15 text-green-400",
      icon: Radio,
    },
    lagged: {
      label: "Lagged — reconnecting",
      cls: "bg-orange-500/15 text-orange-400",
      icon: AlertCircle,
    },
    terminated: {
      label: "Run ended",
      cls: "bg-blue-500/15 text-blue-400",
      icon: CheckCircle2,
    },
    error: {
      label: "Error",
      cls: "bg-red-500/15 text-red-400",
      icon: XCircle,
    },
  };
  const { label, cls, icon: Icon } = map[status];
  return (
    <Badge className={cn("gap-1 font-mono", cls)}>
      <Icon className="h-3 w-3" />
      {label}
    </Badge>
  );
}

// ── Single event row ───────────────────────────────────────────────
function EventRow({ event }: { event: RunEvent }) {
  const cat = categoryForEvent(event.name);
  const meta = CATEGORIES[cat] ?? { label: cat, color: "" };
  const time = new Date(event.receivedAt).toLocaleTimeString(undefined, {
    hour12: false,
    fractionalSecondDigits: 3,
  });
  const summary = useMemo(() => {
    const d = event.data;
    // Produce a short human-readable summary from common payload shapes.
    if (typeof d.tool_name === "string") return String(d.tool_name);
    if (typeof d.status === "string") return String(d.status);
    if (typeof d.delta === "string") {
      return d.delta.length > 80 ? d.delta.slice(0, 77) + "…" : d.delta;
    }
    if (typeof d.message === "string") return d.message;
    if (typeof d.reasoning === "string") {
      return d.reasoning.length > 80
        ? d.reasoning.slice(0, 77) + "…"
        : d.reasoning;
    }
    return "";
  }, [event.data]);

  return (
    <div className="flex items-start gap-2 border-b border-border/40 py-1.5 font-mono text-xs">
      <span className="shrink-0 text-muted-foreground tabular-nums">{time}</span>
      {event.id != null && (
        <span className="shrink-0 text-muted-foreground/60 tabular-nums">
          #{event.id}
        </span>
      )}
      <Badge className={cn("shrink-0", meta.color)}>{meta.label}</Badge>
      <span className="shrink-0 text-foreground/80">{event.name}</span>
      {summary && (
        <span className="min-w-0 truncate text-muted-foreground">{summary}</span>
      )}
    </div>
  );
}

// ── Empty state ────────────────────────────────────────────────────
function EmptyState({
  runId,
  status,
}: {
  runId: string;
  status: RunEventsStatus;
}) {
  if (status === "connecting") {
    return (
      <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
        <Spinner />
        Connecting to run {runId}…
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="flex items-center gap-2 py-8 text-sm text-red-400">
        <XCircle className="h-4 w-4" />
        Could not connect — the api_server may be unavailable.
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
      <Clock className="h-4 w-4" />
      Waiting for events…
    </div>
  );
}

export default function RunsPage() {
  const [runIdInput, setRunIdInput] = useState("");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [selectedCats, setSelectedCats] = useState<Set<string>>(
    new Set(ALL_CATEGORIES),
  );
  const [capabilitiesAvailable, setCapabilitiesAvailable] = useState<
    boolean | null
  >(null);

  // Probe whether the api_server advertises run-events SSE.
  const checkCapabilities = useCallback(async () => {
    try {
      const base =
        (typeof window !== "undefined" &&
          (window as Window & { __HERMES_BASE_PATH__?: string })
            .__HERMES_BASE_PATH__) ||
        "";
      const token =
        typeof window !== "undefined"
          ? (window as Window & { __HERMES_SESSION_TOKEN__?: string })
              .__HERMES_SESSION_TOKEN__
          : undefined;
      const url = `${base}/api/runs/capabilities${token ? `?token=${encodeURIComponent(token)}` : ""}`;
      const headers: HeadersInit = {};
      if (token) headers["X-Hermes-Session-Token"] = token;
      const r = await fetch(url, { headers });
      if (!r.ok) {
        setCapabilitiesAvailable(false);
        return;
      }
      const data = (await r.json()) as { available?: boolean };
      setCapabilitiesAvailable(data.available === true);
    } catch {
      setCapabilitiesAvailable(false);
    }
  }, []);

  // Check capabilities once on mount.
  useMemo(() => {
    void checkCapabilities();
  }, [checkCapabilities]);

  const includeParam = useMemo(() => {
    // Build the include query param from selected categories.
    // The "meta" category is always delivered (mandatory), so we don't send it.
    const cats = ALL_CATEGORIES.filter((c) => selectedCats.has(c));
    return cats.length === ALL_CATEGORIES.length ? undefined : cats.join(",");
  }, [selectedCats]);

  const {
    events,
    capabilities,
    status,
    error,
    lastEventId,
    eventCount,
    clearEvents,
  } = useRunEvents({
    runId: activeRunId,
    include: includeParam,
    enabled: !!activeRunId,
  });

  const handleSubmit = useCallback(() => {
    const trimmed = runIdInput.trim();
    if (trimmed) {
      setActiveRunId(trimmed);
      clearEvents();
    }
  }, [runIdInput, clearEvents]);

  const toggleCategory = useCallback((cat: string) => {
    setSelectedCats((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) {
        if (next.size > 1) next.delete(cat); // never empty
      } else {
        next.add(cat);
      }
      return next;
    });
  }, []);

  return (
    <div className="flex h-full flex-col gap-4 overflow-hidden p-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <H2 className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          Run Events
        </H2>
        {capabilitiesAvailable === false && (
          <Badge className="bg-amber-500/15 text-amber-400">
            api_server not detected
          </Badge>
        )}
      </div>

      {/* Run ID input + connect */}
      <Card>
        <CardContent className="flex items-end gap-3 p-4">
          <div className="flex-1">
            <Label htmlFor="run-id-input" className="mb-1.5 block text-sm">
              Run ID
            </Label>
            <Input
              id="run-id-input"
              value={runIdInput}
              onChange={(e) => setRunIdInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
              }}
              placeholder="e.g. run_abc123 or a UUID"
              className="font-mono"
            />
          </div>
          <Button onClick={handleSubmit} disabled={!runIdInput.trim()}>
            {activeRunId ? (
              <>
                <Zap className="mr-1 h-4 w-4" />
                Switch
              </>
            ) : (
              <>
                <Play className="mr-1 h-4 w-4" />
                Connect
              </>
            )}
          </Button>
          {activeRunId && (
            <Button
              outlined
              onClick={() => {
                setActiveRunId(null);
                setRunIdInput("");
              }}
            >
              <XCircle className="mr-1 h-4 w-4" />
              Disconnect
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Capabilities + status bar */}
      {activeRunId && (
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <StatusBadge status={status} />
          <span className="font-mono text-muted-foreground">
            run: <span className="text-foreground">{activeRunId}</span>
          </span>
          {lastEventId != null && (
            <span className="font-mono text-muted-foreground">
              last ID: <span className="text-foreground">#{lastEventId}</span>
            </span>
          )}
          <span className="font-mono text-muted-foreground">
            {eventCount} event{eventCount !== 1 ? "s" : ""}
          </span>
          {error && (
            <span className="font-mono text-red-400">{error}</span>
          )}
          {capabilities && (
            <span className="font-mono text-muted-foreground/70">
              v{capabilities.version} · {capabilities.snapshot_id ?? "—"} ·{" "}
              heartbeat {capabilities.heartbeat_seconds ?? "?"}s
            </span>
          )}
        </div>
      )}

      {/* Category filters */}
      {activeRunId && (
        <div className="flex flex-wrap items-center gap-1.5">
          {ALL_CATEGORIES.map((cat) => {
            const meta = CATEGORIES[cat];
            const active = selectedCats.has(cat);
            return (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                className={cn(
                  "rounded-md border px-2 py-0.5 text-xs font-medium transition-colors",
                  active
                    ? cn(meta.color, "border-transparent")
                    : "border-border text-muted-foreground hover:bg-muted",
                )}
              >
                {meta.label}
              </button>
            );
          })}
          <Button
            ghost
            size="sm"
            className="ml-auto h-7 px-2 text-xs"
            onClick={clearEvents}
          >
            <Trash2 className="mr-1 h-3 w-3" />
            Clear
          </Button>
        </div>
      )}

      {/* Event feed */}
      {activeRunId ? (
        <Card className="flex-1 overflow-hidden">
          <CardContent className="h-full overflow-y-auto p-3">
            {events.length === 0 ? (
              <EmptyState runId={activeRunId} status={status} />
            ) : (
              <div>
                {events.map((ev, i) => (
                  <EventRow
                    key={`${ev.id ?? "noid"}-${i}`}
                    event={ev}
                  />
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      ) : (
        <Card className="flex flex-1 items-center justify-center">
          <CardContent className="flex flex-col items-center gap-3 py-12 text-center">
            <Terminal className="h-10 w-10 text-muted-foreground/50" />
            <div className="text-sm text-muted-foreground">
              Enter a run ID above to start streaming live events.
            </div>
            <div className="flex items-center gap-1 text-xs text-muted-foreground/70">
              <ArrowRight className="h-3 w-3" />
              Events stream from the api_server via SSE.
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
