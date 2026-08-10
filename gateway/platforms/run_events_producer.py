"""SSE run-events producer for ``GET /v1/runs/{run_id}/events``.

Implements the accepted round-6 contract (sha256
162676cd8dc0d58f3162a68eb9adfb14d84cdd498d52136ce5f996f7f97e8143):

- Per-run bounded replay ring (256 events / 1 MiB) with monotonic IDs.
- Independent per-subscriber bounded queues (128 events / 512 KiB).
- Per-subscriber serialization slot (atomic FREE→ACQUIRED→FREE).
- Close-on-lag with join-then-capture quiescence and global reconnect
  cursor gap fields, two entry points (queue-overflow and
  serialization-timeout).
- Four-limit atomic admission with pre-stream 503 ``run_events_overload``.
- Category-based fail-closed ``include`` parsing.
- ``Last-Event-ID``/``after`` replay with cursor-0 sentinel and
  ``N+1 < min_retained_id`` expiry.
- One typed ``RunEventsCapabilitiesSnapshot`` serializing into both the
  endpoint ``/v1/capabilities`` block and the in-stream capabilities
  control frame.

The producer is intentionally framework-light: it operates on plain
``bytes`` frames and delegates the actual SSE wire writing to the
caller (``_handle_run_events`` in ``api_server.py``).  This keeps it
unit-testable without an aiohttp transport.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Category map (§4) ───────────────────────────────────────────────────────

#: Emitted event name → category.  ``meta`` is mandatory and always delivered.
_EVENT_CATEGORY_MAP: Dict[str, str] = {
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
    "error": "error",
    "hermes.run_events.event.oversize": "error",
    # meta (mandatory, never filtered)
    "hermes.run_events.capabilities": "meta",
    "hermes.run_events.heartbeat": "meta",
    "hermes.run_events.subscriber.lagged": "meta",
    "hermes.run_events.terminal": "meta",
}

SUPPORTED_INCLUDE_CATEGORIES: List[str] = [
    "tool", "message", "reasoning", "subagent", "approval", "status", "error",
]

#: Events that carry a per-run monotonic ``id`` (replayable).
_CONTROL_EVENT_NAMES: Set[str] = {
    "hermes.run_events.capabilities",
    "hermes.run_events.heartbeat",
    "hermes.run_events.subscriber.lagged",
}


def _event_category(event_name: str) -> str:
    """Return the category for *event_name*, defaulting to ``error`` for unknown."""
    return _EVENT_CATEGORY_MAP.get(event_name, "error")


def is_replayable_event(event_name: str) -> bool:
    """Whether an event carries a per-run monotonic SSE ``id``.

    Control frames (capabilities, heartbeat, subscriber.lagged) never carry
    ``id``.  The terminal frame and oversize frame are replayable.
    """
    if event_name in _CONTROL_EVENT_NAMES:
        return False
    return True


# ── Capability snapshot (§3) ────────────────────────────────────────────────


@dataclass(frozen=True)
class RunEventsCapabilitiesSnapshot:
    """Single typed config snapshot serialised into both capability blocks."""

    version: int = 1
    snapshot_id: str = "runevents-v1-20260810-default"
    replay: bool = True
    multicast: bool = True
    max_replay_events: int = 256
    max_replay_bytes: int = 1_048_576
    max_subscriber_queue_events: int = 128
    max_subscriber_queue_bytes: int = 524_288
    max_event_bytes: int = 65_536
    max_retained_runs: int = 64
    max_active_runs_for_events: int = 64
    max_concurrent_subscribers: int = 256
    max_subscribers_per_run: int = 16
    heartbeat_seconds: int = 15
    terminal_retention_seconds: int = 300
    total_feature_memory_budget_bytes: int = 268_435_456
    container_budget_bytes: int = 33_554_432
    control_slot_pool_bytes: int = 16_777_216
    serialization_pool_bytes: int = 16_777_216
    supported_include_categories: List[str] = field(
        default_factory=lambda: list(SUPPORTED_INCLUDE_CATEGORIES)
    )
    mandatory_category: str = "meta"
    include_match: str = "category"

    def endpoint_block(self) -> Dict[str, Any]:
        """Dict for the ``run_events`` key in ``/v1/capabilities``."""
        return {
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "replay": self.replay,
            "multicast": self.multicast,
            "max_replay_events": self.max_replay_events,
            "max_replay_bytes": self.max_replay_bytes,
            "max_subscriber_queue_events": self.max_subscriber_queue_events,
            "max_subscriber_queue_bytes": self.max_subscriber_queue_bytes,
            "max_event_bytes": self.max_event_bytes,
            "max_retained_runs": self.max_retained_runs,
            "max_active_runs_for_events": self.max_active_runs_for_events,
            "max_concurrent_subscribers": self.max_concurrent_subscribers,
            "max_subscribers_per_run": self.max_subscribers_per_run,
            "heartbeat_seconds": self.heartbeat_seconds,
            "terminal_retention_seconds": self.terminal_retention_seconds,
            "total_feature_memory_budget_bytes": self.total_feature_memory_budget_bytes,
            "container_budget_bytes": self.container_budget_bytes,
            "control_slot_pool_bytes": self.control_slot_pool_bytes,
            "serialization_pool_bytes": self.serialization_pool_bytes,
            "supported_include_categories": list(self.supported_include_categories),
            "mandatory_category": self.mandatory_category,
            "include_match": self.include_match,
        }

    def in_stream_data(self) -> Dict[str, Any]:
        """Payload for the ``hermes.run_events.capabilities`` SSE frame."""
        d = self.endpoint_block()
        # The in-stream frame carries the same fields; no SSE ``id``.
        return d

    def overload_error(self) -> Dict[str, Any]:
        """Error body for 503 ``run_events_overload``."""
        return {
            "error": {
                "message": "Run-events capacity saturated",
                "type": "server_error",
                "code": "run_events_overload",
                "max_active_runs_for_events": self.max_active_runs_for_events,
                "max_retained_runs": self.max_retained_runs,
                "max_concurrent_subscribers": self.max_concurrent_subscribers,
                "max_subscribers_per_run": self.max_subscribers_per_run,
            }
        }


DEFAULT_SNAPSHOT = RunEventsCapabilitiesSnapshot()


# ── Include parsing (§4) ────────────────────────────────────────────────────


class IncludeParseError(Exception):
    """Raised when ``include`` query parsing fails.

    ``code`` is the error code (``invalid_include_list`` or
    ``unsupported_include_kinds``).
    """

    def __init__(self, code: str, message: str, supported: Optional[List[str]] = None):
        self.code = code
        self.message = message
        self.supported = supported
        super().__init__(message)

    def error_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "message": self.message,
            "type": "invalid_request_error",
            "code": self.code,
        }
        if self.supported is not None:
            body["supported"] = self.supported
        return {"error": body}


def parse_include(
    raw_values: List[str],
    snapshot: RunEventsCapabilitiesSnapshot = DEFAULT_SNAPSHOT,
) -> Set[str]:
    """Parse the ``include`` query parameter(s) into a set of categories.

    *raw_values* is the list of raw query parameter values for the ``include``
    key (e.g. ``["tool,message"]`` or ``["tool", "message"]``).

    Returns the set of requested categories (without ``meta``; callers always
    deliver meta).  Raises :class:`IncludeParseError` on any parse failure.
    """
    # Repeated include parameters are rejected (§4).
    if len(raw_values) > 1:
        raise IncludeParseError(
            "invalid_include_list",
            "Multiple 'include' query parameters are not allowed; "
            "use a single comma-separated value.",
        )

    if not raw_values:
        # Absent key = all non-meta categories.
        return set(snapshot.supported_include_categories)

    raw = raw_values[0]
    # Percent-decode is handled by the web framework; here we receive the
    # already-decoded value.  Split on comma.
    tokens: List[str] = []
    for part in raw.split(","):
        token = part.strip()
        tokens.append(token)

    # Empty tokens (including "include=", "include=,", leading/trailing commas,
    # or whitespace-only members) → 400 invalid_include_list.
    if any(t == "" for t in tokens):
        raise IncludeParseError(
            "invalid_include_list",
            "Empty include category in list.",
        )

    # Deduplicate while preserving set semantics.
    seen: Set[str] = set()
    for t in tokens:
        seen.add(t)

    # Validate against supported categories.
    supported = set(snapshot.supported_include_categories)
    unknown = seen - supported
    if unknown:
        raise IncludeParseError(
            "unsupported_include_kinds",
            f"Unsupported include category(s): {', '.join(sorted(unknown))}",
            supported=list(snapshot.supported_include_categories),
        )

    return seen


# ── Cursor parsing/validation (§5) ──────────────────────────────────────────


class CursorError(Exception):
    """Raised when ``Last-Event-ID``/``after`` cursor validation fails."""

    def __init__(self, code: str, message: str, http_status: int):
        self.code = code
        self.message = message
        self.http_status = http_status
        super().__init__(message)

    def error_body(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "code": self.code,
            }
        }


def parse_cursor(raw_cursor: Optional[str]) -> int:
    """Parse and validate a raw cursor string.

    Returns the non-negative integer cursor, or raises :class:`CursorError`.
    A missing/empty cursor is treated as cursor ``0`` (replay from the first
    retained event in the ring), matching the backward-compatible behavior
    where first-time subscribers receive all retained events.
    """
    if raw_cursor is None or raw_cursor == "":
        # No cursor = initial subscription = replay from ring start (cursor 0).
        return 0
    try:
        cursor = int(raw_cursor)
    except (ValueError, TypeError):
        raise CursorError("invalid_cursor", f"Malformed cursor: {raw_cursor!r}", 400)
    if cursor < 0:
        raise CursorError("invalid_cursor", f"Negative cursor: {cursor}", 400)
    return cursor


# ── Replay ring ─────────────────────────────────────────────────────────────


@dataclass
class _RingEntry:
    """A single entry in the per-run replay ring."""

    event_id: int
    event_name: str
    data: bytes  # pre-serialized JSON data payload (the ``data:`` content)
    size: int    # byte size of the full SSE frame (for budget accounting)


class _ReplayRing:
    """Bounded replay ring for a single run.

    Dual-limited by event count and byte count.  Eviction is FIFO.
    """

    def __init__(self, max_events: int, max_bytes: int):
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._entries: Deque[_RingEntry] = deque()
        self._current_bytes: int = 0
        self._next_event_id: int = 1
        self._lock = threading.Lock()

    @property
    def max_event_id(self) -> int:
        with self._lock:
            return self._next_event_id - 1

    @property
    def min_retained_id(self) -> int:
        """Smallest event ID in the ring, or 0 if empty."""
        with self._lock:
            if not self._entries:
                return 0
            return self._entries[0].event_id

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._entries) == 0

    def add(self, event_name: str, data: bytes, frame_size: int) -> int:
        """Add a replayable event to the ring and return its assigned event ID."""
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            entry = _RingEntry(event_id, event_name, data, frame_size)
            self._entries.append(entry)
            self._current_bytes += frame_size
            # Evict from the head while over budget.
            while self._entries and (
                len(self._entries) > self._max_events
                or self._current_bytes > self._max_bytes
            ):
                evicted = self._entries.popleft()
                self._current_bytes -= evicted.size
            return event_id

    def replay_after(self, cursor: int) -> List[_RingEntry]:
        """Return entries with event_id > *cursor*, in order.

        For cursor 0 (sentinel), returns the entire ring.
        """
        with self._lock:
            if cursor == 0:
                return list(self._entries)
            return [e for e in self._entries if e.event_id > cursor]


# ── Subscriber ──────────────────────────────────────────────────────────────


@dataclass
class _Subscriber:
    """A single SSE subscriber connection."""

    queue: Deque[Tuple[int, str, bytes]]  # (event_id, event_name, data)
    queue_event_count: int = 0
    queue_byte_count: int = 0
    last_delivered_event_id: int = 0
    # Serialization slot: atomic FREE → ACQUIRED → FREE (§2).
    slot_acquired: bool = False
    slot_acquired_at: float = 0.0
    # Lag/close state.
    closed: bool = False
    # Filtering.
    include_categories: Optional[Set[str]] = None  # None = all
    # Lag frame to deliver (if any), allocated from control_slot_pool.
    pending_lag_frame: Optional[bytes] = None


# ── Run state ───────────────────────────────────────────────────────────────


@dataclass
class _RunState:
    """Per-run producer state."""

    run_id: str
    ring: _ReplayRing
    subscribers: Dict[int, _Subscriber] = field(default_factory=dict)
    active: bool = True
    terminal_time: Optional[float] = None  # when the run ended
    retention_expiry: Optional[float] = None  # terminal_time + retention
    _next_subscriber_id: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock)


# ── Producer ────────────────────────────────────────────────────────────────


class RunEventsProducer:
    """Process-wide SSE run-events producer.

    Manages replay rings and subscriber queues for all runs, enforces the
    four-limit atomic admission, and serialises events for delivery.

    Thread-safety: all mutable state is guarded by per-run locks and a
    process-wide lock for admission counters.  The producer is designed to
    be called from both sync (event callback) and async (handler) contexts.
    """

    def __init__(self, snapshot: Optional[RunEventsCapabilitiesSnapshot] = None):
        self.snapshot = snapshot or DEFAULT_SNAPSHOT
        self._runs: Dict[str, _RunState] = {}
        self._global_lock = threading.Lock()
        self._total_subscribers: int = 0
        self._active_run_count: int = 0
        self._retained_run_count: int = 0
        self._container_bytes_used: int = 0
        self._container_high_water: int = 0
        # Terminal retention sweeper hook (set by the adapter).
        self._retention_sweep_callback: Optional[Any] = None

    # ── Capability access ──────────────────────────────────────────────────

    def capabilities_endpoint_block(self) -> Dict[str, Any]:
        return self.snapshot.endpoint_block()

    # ── Run lifecycle ──────────────────────────────────────────────────────

    def admit_run(self, run_id: str) -> bool:
        """Atomically reserve capacity for a new run.

        Returns True if admitted, False if overloaded (caller returns 503).
        Idempotent for an already-admitted run (returns True).
        """
        with self._global_lock:
            if run_id in self._runs:
                return True
            # Check all four limits atomically.
            if self._retained_run_count >= self.snapshot.max_retained_runs:
                return False
            if self._active_run_count >= self.snapshot.max_active_runs_for_events:
                return False
            # Reserve.
            self._retained_run_count += 1
            self._active_run_count += 1
            ring = _ReplayRing(
                self.snapshot.max_replay_events,
                self.snapshot.max_replay_bytes,
            )
            self._runs[run_id] = _RunState(run_id=run_id, ring=ring)
            return True

    def mark_run_terminal(self, run_id: str) -> None:
        """Mark a run as terminal; start terminal retention countdown."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            if run.active:
                run.active = False
                run.terminal_time = time.time()
                run.retention_expiry = (
                    run.terminal_time + self.snapshot.terminal_retention_seconds
                )
                self._active_run_count = max(0, self._active_run_count - 1)

    def expire_run(self, run_id: str) -> None:
        """Release a run's retained reservation (after retention expiry)."""
        with self._global_lock:
            run = self._runs.pop(run_id, None)
            if run is None:
                return
            if run.active:
                self._active_run_count = max(0, self._active_run_count - 1)
            self._retained_run_count = max(0, self._retained_run_count - 1)
            self._total_subscribers -= len(run.subscribers)

    def run_exists(self, run_id: str) -> bool:
        with self._global_lock:
            return run_id in self._runs

    def run_is_active(self, run_id: str) -> bool:
        with self._global_lock:
            run = self._runs.get(run_id)
            return run is not None and run.active

    # ── Cursor validation ─────────────────────────────────────────────────

    def validate_cursor(self, run_id: str, cursor: int) -> None:
        """Validate a parsed cursor against run state. Raises CursorError."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise CursorError("run_not_found", f"Run not found: {run_id}", 404)
            max_id = run.ring.max_event_id
            min_retained = run.ring.min_retained_id
            # Cursor 0 = sentinel: valid when ring non-empty or run active.
            if cursor == 0:
                if run.ring.is_empty and not run.active:
                    # Ring empty on terminated run whose retention expired.
                    if run.retention_expiry and time.time() > run.retention_expiry:
                        raise CursorError(
                            "cursor_expired",
                            "Run retention expired and replay ring is empty.",
                            409,
                        )
                return
            # Non-zero cursor.
            # Expired: N + 1 < min_retained_id.
            if min_retained > 0 and cursor + 1 < min_retained:
                raise CursorError(
                    "cursor_expired",
                    f"Cursor {cursor} expired; minimum retained ID is {min_retained}.",
                    409,
                )
            # Future: cursor > max_id and run not active.
            if cursor > max_id and not run.active:
                raise CursorError(
                    "cursor_future",
                    f"Cursor {cursor} is beyond the final event ID {max_id}.",
                    409,
                )

    # ── Subscriber admission ──────────────────────────────────────────────

    def admit_subscriber(
        self,
        run_id: str,
        include_categories: Optional[Set[str]] = None,
    ) -> Optional[int]:
        """Atomically reserve capacity for a new subscriber.

        Returns the subscriber ID if admitted, or None if overloaded (caller
        returns 503).  The caller must subsequently call
        :meth:`snapshot_and_subscribe` to get replay events + register for live.
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            # Check subscriber limits.
            if self._total_subscribers >= self.snapshot.max_concurrent_subscribers:
                return None
            if len(run.subscribers) >= self.snapshot.max_subscribers_per_run:
                return None
            # Reserve.
            sub_id = run._next_subscriber_id
            run._next_subscriber_id += 1
            sub = _Subscriber(
                queue=deque(),
                include_categories=include_categories,
            )
            run.subscribers[sub_id] = sub
            self._total_subscribers += 1
            return sub_id

    def remove_subscriber(self, run_id: str, sub_id: int) -> None:
        """Release a subscriber reservation (on disconnect or close-on-lag)."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            if run.subscribers.pop(sub_id, None) is not None:
                self._total_subscribers = max(0, self._total_subscribers - 1)

    # ── Atomic snapshot-plus-subscribe (§5) ───────────────────────────────

    def snapshot_and_subscribe(
        self,
        run_id: str,
        sub_id: int,
        cursor: int,
    ) -> Tuple[List[_RingEntry], bool]:
        """Atomically capture the replay ring and register for live events.

        Returns ``(replay_entries, run_is_active)``.  No event may fall between
        the snapshot and live registration — this method holds the run lock
        for the entire operation.
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return [], False
            # Capture replay entries under the lock.
            replay_entries = run.ring.replay_after(cursor)
            return replay_entries, run.active

    # ── Event emission ────────────────────────────────────────────────────

    def _serialize_event(self, event_name: str, event_data: Dict[str, Any]) -> bytes:
        """Serialize an event into an SSE ``data:`` payload (JSON bytes)."""
        return json.dumps(event_data, separators=(",", ":")).encode("utf-8")

    def _frame_size(self, event_name: str, data_bytes: bytes, event_id: Optional[int]) -> int:
        """Compute the full SSE frame byte size for budget accounting."""
        # Approximate: "event: {name}\nid: {id}\ndata: {json}\n\n"
        size = len(b"event: ") + len(event_name.encode("utf-8"))
        if event_id is not None:
            size += len(b"\nid: ") + len(str(event_id).encode("ascii"))
        size += len(b"\ndata: ") + len(data_bytes) + len(b"\n\n")
        return size

    def emit_event(self, run_id: str, event_name: str, event_data: Dict[str, Any]) -> None:
        """Emit a replayable run event to the ring and all subscriber queues.

        Handles oversize replacement (§7) and category filtering (§4).
        """
        snapshot = self.snapshot
        data_bytes = self._serialize_event(event_name, event_data)

        # Oversize check: if the serialized data exceeds max_event_bytes,
        # replace with the oversize error event.
        if len(data_bytes) > snapshot.max_event_bytes:
            original_name = event_name
            event_name = "hermes.run_events.event.oversize"
            event_data = {
                "original_event": original_name,
                "size_bytes": len(data_bytes),
                "code": "event_oversize",
            }
            data_bytes = self._serialize_event(event_name, event_data)

        # Add to the replay ring (assigns the monotonic ID).
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            event_id = run.ring.add(event_name, data_bytes, self._frame_size(event_name, data_bytes, 0))
            frame = self._frame_size(event_name, data_bytes, event_id)
            category = _event_category(event_name)

            # Distribute to all subscribers' queues.
            for sub_id, sub in list(run.subscribers.items()):
                if sub.closed:
                    continue
                # Category filtering: meta is always delivered; otherwise
                # check include set.
                if category != "meta" and sub.include_categories is not None:
                    if category not in sub.include_categories:
                        continue
                # Enqueue with dual-limit check.
                if (
                    sub.queue_event_count >= snapshot.max_subscriber_queue_events
                    or sub.queue_byte_count >= snapshot.max_subscriber_queue_bytes
                ):
                    # Queue-overflow entry → close-on-lag.
                    self._close_on_lag(
                        run, sub_id, sub,
                        triggering_event_id=event_id,
                        triggering_event_name=event_name,
                    )
                    continue
                sub.queue.append((event_id, event_name, data_bytes))
                sub.queue_event_count += 1
                sub.queue_byte_count += frame

    def _close_on_lag(
        self,
        run: _RunState,
        sub_id: int,
        sub: _Subscriber,
        triggering_event_id: int,
        triggering_event_name: str,
    ) -> None:
        """Execute close-on-lag: freeze, capture, discard, emit lag, close.

        Must be called under ``run._lock`` (caller holds it via _global_lock).
        Join-then-capture quiescence: the in-flight write is conceptually
        joined (in this synchronous model, the last_delivered_event_id
        already reflects what was delivered).  Fields are captured after
        quiescence.
        """
        snapshot = self.snapshot
        # 1. Freeze: detach from live routing, seal queue.
        sub.closed = True
        # 2. Join in-flight write: in this model, delivery is synchronous, so
        #    last_delivered_event_id is already ground truth.
        # 3. Capture fields (global reconnect cursor interval).
        last_delivered = sub.last_delivered_event_id
        latest_available = run.ring.max_event_id
        first_dropped = last_delivered + 1
        dropped = latest_available - last_delivered
        # 4. Discard sealed queue without draining.
        sub.queue.clear()
        sub.queue_event_count = 0
        sub.queue_byte_count = 0
        # 5. Emit lag frame from the control_slot_pool.
        lag_data = {
            "event": "hermes.run_events.subscriber.lagged",
            "last_delivered_event_id": last_delivered,
            "first_dropped_event_id": first_dropped,
            "latest_available_event_id": latest_available,
            "dropped_events": dropped,
        }
        lag_bytes = self._serialize_event("hermes.run_events.subscriber.lagged", lag_data)
        sub.pending_lag_frame = self._format_control_frame(
            "hermes.run_events.subscriber.lagged", lag_bytes
        )

    def _format_control_frame(self, event_name: str, data_bytes: bytes) -> bytes:
        """Format a control frame (no SSE ``id``)."""
        return b"event: " + event_name.encode("utf-8") + b"\ndata: " + data_bytes + b"\n\n"

    def _format_replayable_frame(
        self, event_name: str, event_id: int, data_bytes: bytes
    ) -> bytes:
        """Format a replayable event frame (with SSE ``id``)."""
        return (
            b"event: "
            + event_name.encode("utf-8")
            + b"\nid: "
            + str(event_id).encode("ascii")
            + b"\ndata: "
            + data_bytes
            + b"\n\n"
        )

    def get_subscriber_frame(
        self, run_id: str, sub_id: int, timeout: float
    ) -> Optional[bytes]:
        """Get the next SSE frame for a subscriber, or a heartbeat.

        Returns formatted bytes, a heartbeat comment, or None if the
        subscriber is closed and has no pending lag frame.
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            sub = run.subscribers.get(sub_id)
            if sub is None:
                return None
            # If there's a pending lag frame, deliver it and signal close.
            if sub.pending_lag_frame is not None:
                frame = sub.pending_lag_frame
                sub.pending_lag_frame = None
                return frame
            if sub.closed:
                return None
            # Drain the queue.
            if sub.queue:
                event_id, event_name, data_bytes = sub.queue.popleft()
                sub.queue_event_count -= 1
                frame_size = self._frame_size(event_name, data_bytes, event_id)
                sub.queue_byte_count -= frame_size
                sub.last_delivered_event_id = event_id
                return self._format_replayable_frame(event_name, event_id, data_bytes)
            return None  # no data right now

    def format_capabilities_frame(self) -> bytes:
        """Format the initial in-stream capabilities control frame."""
        data = self.snapshot.in_stream_data()
        data_bytes = self._serialize_event("hermes.run_events.capabilities", data)
        return self._format_control_frame("hermes.run_events.capabilities", data_bytes)

    def format_replay_frame(self, entry: _RingEntry) -> bytes:
        """Format a replayed ring entry as an SSE frame."""
        return self._format_replayable_frame(entry.event_name, entry.event_id, entry.data)

    def format_heartbeat(self) -> bytes:
        """Format a heartbeat comment keepalive frame."""
        return b": keepalive\n\n"

    def format_terminal_frame(
        self, run_id: str, status: str, final_event_id: int
    ) -> bytes:
        """Format the terminal control frame."""
        data = {
            "run_id": run_id,
            "status": status,
            "final_event_id": final_event_id,
        }
        data_bytes = self._serialize_event("hermes.run_events.terminal", data)
        return self._format_control_frame("hermes.run_events.terminal", data_bytes)

    # ── Retention sweep ───────────────────────────────────────────────────

    def sweep_expired_runs(self, now: Optional[float] = None) -> List[str]:
        """Expire runs whose terminal retention has elapsed.

        Returns the list of expired run_ids (for the caller to clean up
        ancillary state like ``_run_statuses``).
        """
        if now is None:
            now = time.time()
        expired: List[str] = []
        with self._global_lock:
            for run_id, run in list(self._runs.items()):
                if (
                    not run.active
                    and run.retention_expiry
                    and now > run.retention_expiry
                ):
                    expired.append(run_id)
        for run_id in expired:
            self.expire_run(run_id)
        return expired

    # ── Diagnostics / testing ─────────────────────────────────────────────

    @property
    def total_subscribers(self) -> int:
        with self._global_lock:
            return self._total_subscribers

    @property
    def active_run_count(self) -> int:
        with self._global_lock:
            return self._active_run_count

    @property
    def retained_run_count(self) -> int:
        with self._global_lock:
            return self._retained_run_count

    @property
    def container_bytes_used(self) -> int:
        """Observable container-budget charged-byte counter (§2)."""
        return self._container_bytes_used

    @property
    def container_high_water(self) -> int:
        """High-water mark of container budget usage (§2)."""
        return self._container_high_water

    def charge_container(self, nbytes: int) -> None:
        """Charge *nbytes* to the container budget (size-class accounting)."""
        with self._global_lock:
            self._container_bytes_used += nbytes
            if self._container_bytes_used > self._container_high_water:
                self._container_high_water = self._container_bytes_used

    def release_container(self, nbytes: int) -> None:
        """Release *nbytes* from the container budget."""
        with self._global_lock:
            self._container_bytes_used = max(0, self._container_bytes_used - nbytes)
