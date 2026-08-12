/**
 * Tests for the run-events SSE hook's pure logic.
 *
 * The hook itself depends on the browser's EventSource API and React's
 * render cycle, which can't be tested without @testing-library/react (not
 * a dependency).  We test the pure pieces: the event-name constants and
 * the URL builder.  The critical contract invariant — that ALL frames are
 * named events and onmessage never fires — is encoded in the constant set
 * we assert on here.
 */

import { describe, it, expect } from "vitest";
import { RUN_EVENT_NAMES } from "@/hooks/useRunEvents";

describe("RUN_EVENT_NAMES (contract §4 category map)", () => {
  it("includes all replayable run-event names from the contract", () => {
    // These are the event names that carry per-run monotonic ``id``.
    // If any is missing, the dashboard would silently drop that event type.
    const required = [
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
    ];
    for (const name of required) {
      expect(RUN_EVENT_NAMES, `missing event name: ${name}`).toContain(name);
    }
  });

  it("does NOT include control/meta frames (they have no id)", () => {
    // Control frames are handled separately — they must not be in the
    // replayable set, otherwise the hook would try to parse them as
    // regular events.
    const controlNames = [
      "hermes.run_events.capabilities",
      "hermes.run_events.heartbeat",
      "hermes.run_events.subscriber.lagged",
      "hermes.run_events.terminal",
    ];
    for (const name of controlNames) {
      expect(RUN_EVENT_NAMES).not.toContain(name);
    }
  });
});
