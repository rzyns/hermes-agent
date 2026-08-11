"""SSE run-events producer for ``GET /v1/runs/{run_id}/events``.

Implements the ADOPTED contract amendment A1-r4 (sha256
0d4378df38d836780d4b647da2276776a71cc5e570b1ac524261a76c24f30bfb,
version 2026-08-11-r6-A1-r4):

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

Memory model (§2 A1-r4): **dual-ledger enforced byte-accounting budget**.
Two independent ledgers per class: a **reservation ledger** (admission-
time capacity holds that enforce the 256 MiB total) and a **settled-
current ledger** (actual bytes held at any moment, for observability).
Every charge goes through two distinct phases — reserve at admission,
settle at emission — and two distinct release transitions — per-item
settle release and lifetime reservation release.

Delivery model
--------------
``get_subscriber_frame`` dequeues an event and *acquires* the
serialization slot, returning the formatted frame **without** advancing
``last_delivered_event_id``.  The caller MUST call ``confirm_delivered``
after the transport write succeeds (advancing the cursor) or
``mark_write_failed`` if it fails (releasing the slot without advancing).
This ensures ``last_delivered_event_id`` always reflects transport
ground truth — the core invariant for join-then-capture lag recovery.
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

# ── Constants from contract §2 (normative — no deviation permitted) ──────────

#: Normative maximum SSE framing overhead per event (§2).
MAX_SSE_FRAMING_BYTES = 128

#: Container-budget fixed per-object charges (§2, container charge table).
_CONTAINER_CHARGE_RUN = 1_024           # Run container per run
_CONTAINER_CHARGE_SUBSCRIBER = 1_024    # Subscriber container per subscriber
_CONTAINER_CHARGE_MAPPING_ENTRY = 128   # Subscriber mapping-table entry
_CONTAINER_CHARGE_LOCK_STATE = 128      # Lock/state object per subscriber

# Total per-subscriber container charge (subscriber + mapping + lock/state).
_SUBSCRIBER_CONTAINER_TOTAL = (
    _CONTAINER_CHARGE_SUBSCRIBER
    + _CONTAINER_CHARGE_MAPPING_ENTRY
    + _CONTAINER_CHARGE_LOCK_STATE
)

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
    control_frame_budget_bytes: int = 16_777_216
    serialization_budget_bytes: int = 16_777_216
    supported_include_categories: List[str] = field(
        default_factory=lambda: list(SUPPORTED_INCLUDE_CATEGORIES)
    )
    mandatory_category: str = "meta"
    include_match: str = "category"

    @property
    def retained_run_budget(self) -> int:
        return self.max_retained_runs * self.max_replay_bytes

    @property
    def subscriber_queue_budget(self) -> int:
        return self.max_concurrent_subscribers * self.max_subscriber_queue_bytes

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
            "control_frame_budget_bytes": self.control_frame_budget_bytes,
            "serialization_budget_bytes": self.serialization_budget_bytes,
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


# ── Dual-ledger byte-accounting budget (§2 A1-r4) ───────────────────────────
#
# Two independent ledgers per class:
#
# 1. **Reservation ledger** — admission-time capacity holds.  Each class has a
#    per-class ``reserved_<class>_bytes`` counter and a ``total_reserved_bytes``
#    counter.  The reservation ledger enforces the budget bound.
#
# 2. **Settled-current ledger** — actual bytes held at any moment.  Each class
#    has a per-class ``current_<class>_charged_bytes`` counter and a
#    ``total_feature_charged_bytes`` counter.  Provides observability of real
#    usage.
#
# Both ledgers maintain high-water counters.  All operations are atomic under
# the producer's ``_global_lock``.

# Budget class identifiers (internal keys).
_BUDGET_CLASSES = (
    "retained_run",
    "subscriber_queue",
    "control_frame",
    "serialization",
    "container",
)


@dataclass
class _BudgetClassLedgers:
    """Reservation and settled-current ledgers for one budget class."""

    # Reservation ledger.
    reserved_bytes: int = 0
    reserved_high_water: int = 0
    # Settled-current ledger.
    current_charged_bytes: int = 0
    current_high_water: int = 0

    def reserve(self, amount: int, class_budget: int) -> bool:
        """Atomically reserve *amount* against class_budget.

        Updates the reservation-ledger counter and high-water.  Does NOT
        touch the settled-current ledger (container is the exception,
        handled via :meth:`reserve_and_settle_container`).

        Returns True if the reservation fits within class_budget, False
        otherwise.
        """
        assert amount >= 0
        new_reserved = self.reserved_bytes + amount
        if new_reserved > class_budget:
            return False
        self.reserved_bytes = new_reserved
        self.reserved_high_water = max(self.reserved_high_water, self.reserved_bytes)
        return True

    def release_reservation(self, amount: int) -> None:
        """Release a lifetime reservation (decrements reservation ledger)."""
        assert amount >= 0
        self.reserved_bytes = max(0, self.reserved_bytes - amount)

    def settle(self, amount: int) -> None:
        """Atomically settle *amount* within a pre-existing reservation.

        Updates the settled-current ledger counter and high-water.  Never
        touches the reservation ledger.  Per §2 "Atomic settlement (Phase 2)".
        """
        assert amount >= 0
        self.current_charged_bytes += amount
        self.current_high_water = max(self.current_high_water, self.current_charged_bytes)

    def settle_release(self, amount: int) -> None:
        """Release a per-item settled charge (decrements settled-current).

        High-water is monotonic — never decremented.  Per §2 "Settle release".
        """
        assert amount >= 0
        self.current_charged_bytes = max(0, self.current_charged_bytes - amount)

    def reserve_and_settle(self, amount: int, class_budget: int) -> bool:
        """Container dual-ledger atomic admission: reserve + settle in one op.

        For the container class, the charge is final at admission (fixed
        size-class), so Phase 1 and Phase 2 coincide.  This atomically updates
        both ledgers, both high-waters.  Per §2 "Container fixed-charge
        admission path".
        """
        assert amount >= 0
        new_reserved = self.reserved_bytes + amount
        if new_reserved > class_budget:
            return False
        self.reserved_bytes = new_reserved
        self.reserved_high_water = max(self.reserved_high_water, self.reserved_bytes)
        self.current_charged_bytes += amount
        self.current_high_water = max(self.current_high_water, self.current_charged_bytes)
        return True

    def release_container(self, amount: int) -> None:
        """Container release: decrements both ledgers atomically.

        Because the container charge is fixed at admission, both the
        reservation and the settled-current charge are released together.
        """
        assert amount >= 0
        self.reserved_bytes = max(0, self.reserved_bytes - amount)
        self.current_charged_bytes = max(0, self.current_charged_bytes - amount)


@dataclass
class _DualLedgerBudget:
    """Process-wide dual-ledger enforced byte-accounting budget (§2 A1-r4).

    Tracks all five budget classes with reservation and settled-current
    ledgers, plus a total for each.  All operations are performed under the
    producer's ``_global_lock`` (the caller holds it).
    """

    classes: Dict[str, _BudgetClassLedgers] = field(
        default_factory=lambda: {c: _BudgetClassLedgers() for c in _BUDGET_CLASSES}
    )
    # Totals.
    total_reserved: int = 0
    total_reserved_high_water: int = 0
    total_feature_charged: int = 0
    total_feature_high_water: int = 0

    # ── Phase 1: atomic reservation ──────────────────────────────────────

    def reserve(self, cls: str, amount: int, class_budget: int, total_budget: int) -> bool:
        """Atomically reserve *amount* for *cls* against class AND total budget.

        One atomic reserve-and-validate: simultaneously increments the
        per-class reservation ledger and ``total_reserved``.  Checks both
        the per-class budget and the total budget; if either fails, no
        ledger is incremented and returns False.

        Per §2 "Atomic reservation (Phase 1)".
        """
        assert amount >= 0
        if amount == 0:
            return True
        ledger = self.classes[cls]
        new_class_reserved = ledger.reserved_bytes + amount
        new_total = self.total_reserved + amount
        if new_class_reserved > class_budget:
            return False
        if new_total > total_budget:
            return False
        ledger.reserved_bytes = new_class_reserved
        ledger.reserved_high_water = max(ledger.reserved_high_water, ledger.reserved_bytes)
        self.total_reserved = new_total
        self.total_reserved_high_water = max(self.total_reserved_high_water, self.total_reserved)
        return True

    def release_reservation(self, cls: str, amount: int) -> None:
        """Release a lifetime reservation for *cls* (reservation release).

        Decrements the per-class reservation ledger and ``total_reserved``.
        Per §2 "Reservation release (lifetime)".
        """
        assert amount >= 0
        if amount == 0:
            return
        ledger = self.classes[cls]
        ledger.reserved_bytes = max(0, ledger.reserved_bytes - amount)
        self.total_reserved = max(0, self.total_reserved - amount)

    # ── Phase 2: atomic settlement ───────────────────────────────────────

    def settle(self, cls: str, amount: int) -> None:
        """Atomically settle *amount* for *cls* (within a pre-existing reservation).

        One atomic settle: simultaneously increments the per-class
        settled-current counter, ``total_feature_charged``, and both
        affected high-waters.  Never touches the reservation ledger.

        Per §2 "Atomic settlement (Phase 2)".
        """
        assert amount >= 0
        if amount == 0:
            return
        ledger = self.classes[cls]
        ledger.current_charged_bytes += amount
        ledger.current_high_water = max(ledger.current_high_water, ledger.current_charged_bytes)
        self.total_feature_charged += amount
        self.total_feature_high_water = max(self.total_feature_high_water, self.total_feature_charged)

    def settle_release(self, cls: str, amount: int) -> None:
        """Release a per-item settled charge for *cls* (settle release).

        Decrements the per-class settled-current counter and
        ``total_feature_charged``.  High-waters are monotonic — unchanged.

        Per §2 "Settle release (per item)".
        """
        assert amount >= 0
        if amount == 0:
            return
        ledger = self.classes[cls]
        ledger.current_charged_bytes = max(0, ledger.current_charged_bytes - amount)
        self.total_feature_charged = max(0, self.total_feature_charged - amount)

    # ── Container dual-ledger atomic admission ───────────────────────────

    def reserve_and_settle_container(
        self, amount: int, class_budget: int, total_budget: int
    ) -> bool:
        """Container dual-ledger atomic admission (reserve + settle in one op).

        Atomically increments reservation-ledger and settled-current-ledger
        for the container class, plus both totals and all four high-waters.
        Per §2 "Container fixed-charge admission path".
        """
        assert amount >= 0
        if amount == 0:
            return True
        ledger = self.classes["container"]
        new_class_reserved = ledger.reserved_bytes + amount
        new_total = self.total_reserved + amount
        if new_class_reserved > class_budget:
            return False
        if new_total > total_budget:
            return False
        ledger.reserved_bytes = new_class_reserved
        ledger.reserved_high_water = max(ledger.reserved_high_water, ledger.reserved_bytes)
        ledger.current_charged_bytes += amount
        ledger.current_high_water = max(ledger.current_high_water, ledger.current_charged_bytes)
        self.total_reserved = new_total
        self.total_reserved_high_water = max(self.total_reserved_high_water, self.total_reserved)
        self.total_feature_charged += amount
        self.total_feature_high_water = max(self.total_feature_high_water, self.total_feature_charged)
        return True

    def release_container(self, amount: int) -> None:
        """Release a container charge (decrements both ledgers + total).

        Per §2: container release coincides reservation release + settle
        release because the charge is fixed at admission.
        """
        assert amount >= 0
        if amount == 0:
            return
        ledger = self.classes["container"]
        ledger.reserved_bytes = max(0, ledger.reserved_bytes - amount)
        ledger.current_charged_bytes = max(0, ledger.current_charged_bytes - amount)
        self.total_reserved = max(0, self.total_reserved - amount)
        self.total_feature_charged = max(0, self.total_feature_charged - amount)

    # ── Observability ────────────────────────────────────────────────────

    def observability_snapshot(self) -> Dict[str, int]:
        """Return all 24 counters (12 current, 12 high-water) for testing.

        Per §2 "Test interface": all twenty-four counters must be exposed.
        """
        result: Dict[str, int] = {
            "total_reserved_bytes": self.total_reserved,
            "total_reserved_high_water_bytes": self.total_reserved_high_water,
            "total_feature_charged_bytes": self.total_feature_charged,
            "total_feature_high_water_bytes": self.total_feature_high_water,
        }
        for cls in _BUDGET_CLASSES:
            ledger = self.classes[cls]
            result[f"reserved_{cls}_bytes"] = ledger.reserved_bytes
            result[f"reserved_{cls}_high_water_bytes"] = ledger.reserved_high_water
            result[f"current_{cls}_charged_bytes"] = ledger.current_charged_bytes
            result[f"{cls}_high_water_bytes"] = ledger.current_high_water
        return result


# ── Replay ring ─────────────────────────────────────────────────────────────


@dataclass
class _RingEntry:
    """A single entry in the per-run replay ring."""

    event_id: int
    event_name: str
    #: JSON ``data:`` payload bytes (without SSE framing).
    data: bytes
    #: Full SSE frame byte count (payload + framing).
    size: int


class _ReplayRing:
    """Bounded replay ring for a single run.

    Dual-limited by event count and byte count.  Eviction is FIFO.

    The ring's ``_current_bytes`` never exceeds ``max_bytes`` because
    eviction happens *before* the new entry is appended, and the check
    uses ``current + frame_size`` (not ``current`` alone).
    """

    def __init__(self, max_events: int, max_bytes: int):
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._entries: Deque[_RingEntry] = deque()
        self._current_bytes: int = 0
        self._high_water_bytes: int = 0
        self._next_event_id: int = 1
        self._lock = threading.Lock()

    @property
    def max_event_id(self) -> int:
        with self._lock:
            return self._next_event_id - 1

    def peek_next_event_id(self) -> int:
        """Return the ID that will be assigned to the next added event."""
        with self._lock:
            return self._next_event_id

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

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    @property
    def high_water_bytes(self) -> int:
        with self._lock:
            return self._high_water_bytes

    def add(self, event_name: str, data: bytes, frame_size: int) -> Tuple[int, List[int], int]:
        """Add an event, enforcing the ring's byte/event budget before append.

        Returns ``(event_id, [evicted_event_ids], evicted_bytes)``.  The caller
        uses the evicted list and byte total to settle-release the old frame
        charges *before* settling the new entry.

        The byte budget check uses ``current + frame_size`` and evicts
        entries until the new frame fits.  If a single frame is larger
        than ``max_bytes``, it cannot be retained but still consumes its
        global monotonic ID.
        """
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            evicted_ids: List[int] = []
            evicted_bytes = 0
            # A single frame larger than the ring cannot be retained, but
            # still consumes its global monotonic ID.
            if frame_size > self._max_bytes or self._max_events <= 0:
                return event_id, evicted_ids, evicted_bytes
            # Evict until the new frame fits within both event and byte caps.
            while self._entries and (
                len(self._entries) + 1 > self._max_events
                or self._current_bytes + frame_size > self._max_bytes
            ):
                evicted = self._entries.popleft()
                self._current_bytes -= evicted.size
                evicted_ids.append(evicted.event_id)
                evicted_bytes += evicted.size
            entry = _RingEntry(event_id, event_name, data, frame_size)
            self._entries.append(entry)
            self._current_bytes += frame_size
            # The high-water mark never exceeds max_bytes because the
            # eviction loop guarantees current_bytes + frame_size <= max_bytes
            # before append, and current_bytes is always <= max_bytes.
            self._high_water_bytes = max(self._high_water_bytes, self._current_bytes)
            return event_id, evicted_ids, evicted_bytes

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
    #: Queue entries: (event_id, event_name, data_bytes, frame_size).
    queue: Deque[Tuple[int, str, bytes, int]] = field(default_factory=deque)
    queue_event_count: int = 0
    queue_byte_count: int = 0
    last_delivered_event_id: int = 0
    # Serialization slot: atomic FREE → ACQUIRED → FREE (§2).
    slot_acquired: bool = False
    slot_acquired_at: float = 0.0
    # In-flight event held in the serialization slot.
    in_flight_event_id: int = 0
    in_flight_event_name: str = ""
    #: The in-flight frame's settled charge (for settle-release on confirm/fail).
    in_flight_frame_size: int = 0
    # Lag/close state.
    closed: bool = False
    # Filtering.
    include_categories: Optional[Set[str]] = None  # None = all
    # Lag frame to deliver (if any), charged from control_frame_budget.
    pending_lag_frame: Optional[bytes] = None
    #: The settled charge for the pending lag frame (for settle-release).
    pending_lag_frame_size: int = 0
    # A freeze requests recovery; field capture waits until an acquired write
    # joins via confirm_delivered/mark_write_failed.
    lag_pending: bool = False


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

    Thread-safety: all mutable state is guarded by ``_global_lock``.  The
    producer is designed to be called from both sync (event callback) and
    async (handler) contexts.
    """

    def __init__(self, snapshot: Optional[RunEventsCapabilitiesSnapshot] = None):
        self.snapshot = snapshot or DEFAULT_SNAPSHOT
        self._runs: Dict[str, _RunState] = {}
        self._global_lock = threading.Lock()
        # Dual-ledger budget (§2 A1-r4).
        self._budget = _DualLedgerBudget()
        self._total_subscribers: int = 0
        self._active_run_count: int = 0
        self._retained_run_count: int = 0
        # Track per-run and per-subscriber settled charges for clean release.
        self._run_ring_settled: Dict[str, int] = {}  # run_id → settled ring bytes
        self._sub_queue_settled: Dict[Tuple[str, int], int] = {}  # (run, sub) → settled queue bytes
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
        snap = self.snapshot
        with self._global_lock:
            if run_id in self._runs:
                return True
            # Check all four limits atomically.
            if self._retained_run_count >= snap.max_retained_runs:
                return False
            if self._active_run_count >= snap.max_active_runs_for_events:
                return False
            # ── Phase 1 reservations (dual-ledger, §2) ──
            # Reserve retained_run budget: max_replay_bytes per run.
            if not self._budget.reserve(
                "retained_run", snap.max_replay_bytes,
                snap.retained_run_budget, snap.total_feature_memory_budget_bytes,
            ):
                return False
            # Container dual-ledger atomic admission for run metadata.
            if not self._budget.reserve_and_settle_container(
                _CONTAINER_CHARGE_RUN,
                snap.container_budget_bytes,
                snap.total_feature_memory_budget_bytes,
            ):
                # Rollback the retained_run reservation.
                self._budget.release_reservation("retained_run", snap.max_replay_bytes)
                return False
            # All reservations succeeded — construct state.
            self._retained_run_count += 1
            self._active_run_count += 1
            ring = _ReplayRing(snap.max_replay_events, snap.max_replay_bytes)
            self._runs[run_id] = _RunState(run_id=run_id, ring=ring)
            self._run_ring_settled[run_id] = 0
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
        """Release a run's retained reservation (after retention expiry).

        Per §2 "Retention expiry": reservation release + remaining settle
        releases for all ring items.
        """
        snap = self.snapshot
        with self._global_lock:
            run = self._runs.pop(run_id, None)
            if run is None:
                return
            if run.active:
                self._active_run_count = max(0, self._active_run_count - 1)
            self._retained_run_count = max(0, self._retained_run_count - 1)
            # Release all subscriber reservations + settled charges.
            for sub_id, sub in list(run.subscribers.items()):
                self._release_subscriber_reservations_locked(run_id, sub_id, sub)
            self._total_subscribers -= len(run.subscribers)
            # Release ring settled charges.
            ring_settled = self._run_ring_settled.pop(run_id, 0)
            if ring_settled > 0:
                self._budget.settle_release("retained_run", ring_settled)
            # Release run reservations: retained_run reservation + container.
            self._budget.release_reservation("retained_run", snap.max_replay_bytes)
            self._budget.release_container(_CONTAINER_CHARGE_RUN)

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

    # ── Atomic subscriber admission + snapshot (§5) ────────────────────────

    def admit_subscriber_and_snapshot(
        self,
        run_id: str,
        cursor: int,
        include_categories: Optional[Set[str]] = None,
    ) -> Optional[Tuple[int, List[_RingEntry], bool]]:
        """Atomically admit a subscriber AND capture the replay ring snapshot.

        This is the §5 atomic snapshot-plus-subscribe: no event may fall
        between the ring snapshot and live registration.  The subscriber
        is added to live routing *inside* the same lock that captures the
        ring, so events emitted after this call are exclusively live (in
        the subscriber's queue) and never duplicated in replay.

        Returns ``(sub_id, filtered_replay_entries, run_is_active)``, or
        ``None`` if any limit is saturated (caller returns 503).
        """
        snap = self.snapshot
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            # Check subscriber limits.
            if self._total_subscribers >= snap.max_concurrent_subscribers:
                return None
            if len(run.subscribers) >= snap.max_subscribers_per_run:
                return None
            # ── Phase 1 reservations (dual-ledger, §2) ──
            # Reserve subscriber_queue, control_frame, serialization budgets.
            reservation_amounts = [
                ("subscriber_queue", snap.max_subscriber_queue_bytes, snap.subscriber_queue_budget),
                ("control_frame", snap.max_event_bytes, snap.control_frame_budget_bytes),
                ("serialization", snap.max_event_bytes, snap.serialization_budget_bytes),
            ]
            reserved_classes: List[str] = []
            for cls, amount, class_budget in reservation_amounts:
                if not self._budget.reserve(
                    cls, amount, class_budget, snap.total_feature_memory_budget_bytes,
                ):
                    # Rollback already-reserved classes.
                    for prev_cls in reserved_classes:
                        prev_amount = next(a for c, a, _ in reservation_amounts if c == prev_cls)
                        self._budget.release_reservation(prev_cls, prev_amount)
                    return None
                reserved_classes.append(cls)
            # Container dual-ledger atomic admission for subscriber metadata.
            if not self._budget.reserve_and_settle_container(
                _SUBSCRIBER_CONTAINER_TOTAL,
                snap.container_budget_bytes,
                snap.total_feature_memory_budget_bytes,
            ):
                # Rollback subscriber reservations.
                for cls, amount, _ in reservation_amounts:
                    self._budget.release_reservation(cls, amount)
                return None
            # All reservations succeeded — construct subscriber state.
            sub_id = run._next_subscriber_id
            run._next_subscriber_id += 1
            sub = _Subscriber(include_categories=include_categories)
            run.subscribers[sub_id] = sub
            self._total_subscribers += 1
            self._sub_queue_settled[(run_id, sub_id)] = 0
            # Capture replay entries (atomically — same lock).
            replay_entries = run.ring.replay_after(cursor)
            # Apply include filter to replay entries (§4, §6).
            if include_categories is not None:
                replay_entries = [
                    e for e in replay_entries
                    if self._is_category_allowed(e.event_name, include_categories)
                ]
            return sub_id, replay_entries, run.active

    def _release_subscriber_reservations_locked(
        self, run_id: str, sub_id: int, sub: _Subscriber
    ) -> None:
        """Release all budget reservations + settled charges for a subscriber.

        Must be called under ``_global_lock``.  Per §2 "Subscriber disconnect":
        reservation release + remaining settle releases.
        """
        snap = self.snapshot
        # Release remaining settled queue charges.
        queue_settled = self._sub_queue_settled.pop((run_id, sub_id), 0)
        if queue_settled > 0:
            self._budget.settle_release("subscriber_queue", queue_settled)
        # Release remaining serialization slot settled charge (if any).
        if sub.in_flight_frame_size > 0:
            self._budget.settle_release("serialization", sub.in_flight_frame_size)
            sub.in_flight_frame_size = 0
        # Release pending lag frame charge (if any).
        if sub.pending_lag_frame_size > 0:
            self._budget.settle_release("control_frame", sub.pending_lag_frame_size)
            sub.pending_lag_frame_size = 0
        # Release reservations: queue + control + serialization + container.
        self._budget.release_reservation("subscriber_queue", snap.max_subscriber_queue_bytes)
        self._budget.release_reservation("control_frame", snap.max_event_bytes)
        self._budget.release_reservation("serialization", snap.max_event_bytes)
        self._budget.release_container(_SUBSCRIBER_CONTAINER_TOTAL)

    def remove_subscriber(self, run_id: str, sub_id: int) -> None:
        """Release a subscriber reservation (on disconnect or close-on-lag)."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            sub = run.subscribers.pop(sub_id, None)
            if sub is not None:
                self._release_subscriber_reservations_locked(run_id, sub_id, sub)
                self._total_subscribers = max(0, self._total_subscribers - 1)

    # ── Category filtering helper ─────────────────────────────────────────

    @staticmethod
    def _is_category_allowed(
        event_name: str, include_categories: Optional[Set[str]]
    ) -> bool:
        """Whether *event_name* passes the include filter.

        ``meta`` is always delivered.  When *include_categories* is ``None``
        (absent include key), all categories are delivered.
        """
        if include_categories is None:
            return True
        cat = _event_category(event_name)
        if cat == "meta":
            return True
        return cat in include_categories

    # ── Event emission ────────────────────────────────────────────────────

    def _serialize_event(self, event_name: str, event_data: Dict[str, Any]) -> bytes:
        """Serialize an event into an SSE ``data:`` payload (JSON bytes)."""
        return json.dumps(event_data, separators=(",", ":")).encode("utf-8")

    def _frame_size(self, event_name: str, data_bytes: bytes, event_id: Optional[int]) -> int:
        """Compute the full SSE frame byte size for budget accounting.

        The complete SSE wire frame includes:
        - ``event: {name}\\n``
        - optional ``id: {id}\\n``
        - ``data: {json}\\n``
        - terminating ``\\n``
        """
        # Approximate: "event: {name}\\nid: {id}\\ndata: {json}\\n\\n"
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

        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return

            # ── Assign the event ID before any sizing (B2 fix) ──
            # The real event ID is needed for exact wire-frame sizing.  IDs
            # are assigned by the ring; peek now so oversize validation and
            # all charge values use the actual rendered ``id:`` field width.
            event_id = run.ring.peek_next_event_id()

            # Oversize check: if the complete SSE wire frame (with the real
            # event ID) exceeds max_event_bytes, replace with the oversize
            # error event (§2/§7).
            prelim_frame_size = self._frame_size(event_name, data_bytes, event_id)
            if prelim_frame_size > snapshot.max_event_bytes:
                original_name = event_name
                event_name = "hermes.run_events.event.oversize"
                event_data = {
                    "original_event": original_name,
                    "size_bytes": prelim_frame_size,
                    "code": "event_oversize",
                }
                data_bytes = self._serialize_event(event_name, event_data)

            # ── Phase 2: settle ring insertion (B1 fix) ──
            # Compute the real wire frame size with the actual assigned ID.
            ring_frame_size = self._frame_size(event_name, data_bytes, event_id)
            event_id, evicted_ids, evicted_bytes = run.ring.add(
                event_name, data_bytes, ring_frame_size,
            )
            # Settle-release evicted entries BEFORE settling the replacement.
            # This ordering guarantees the settled-current counter never
            # transiently exceeds the reservation ledger, so the settled
            # high-water stays ≤ the reservation high-water at every instant.
            if evicted_bytes > 0:
                self._budget.settle_release("retained_run", evicted_bytes)
            self._budget.settle("retained_run", ring_frame_size)
            self._run_ring_settled[run_id] = (
                self._run_ring_settled.get(run_id, 0) - evicted_bytes + ring_frame_size
            )

            # Compute the frame size WITH event_id for queue accounting
            # (consistent with the ring charge — same actual ID).
            frame = ring_frame_size  # already sized with the real event_id
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
                # Enqueue with dual-limit check (current + candidate, §6).
                if (
                    sub.queue_event_count + 1 > snapshot.max_subscriber_queue_events
                    or sub.queue_byte_count + frame > snapshot.max_subscriber_queue_bytes
                ):
                    # Queue-overflow entry → close-on-lag.
                    self._close_on_lag(
                        run, sub_id, sub,
                        triggering_event_id=event_id,
                        triggering_event_name=event_name,
                        triggering_frame_size=frame,
                    )
                    continue
                # Enqueue and settle the queue charge.
                sub.queue.append((event_id, event_name, data_bytes, frame))
                sub.queue_event_count += 1
                sub.queue_byte_count += frame
                self._budget.settle("subscriber_queue", frame)
                self._sub_queue_settled[(run_id, sub_id)] = (
                    self._sub_queue_settled.get((run_id, sub_id), 0) + frame
                )

    def _close_on_lag(
        self,
        run: _RunState,
        sub_id: int,
        sub: _Subscriber,
        triggering_event_id: int,
        triggering_event_name: str,
        triggering_frame_size: int = 0,
    ) -> None:
        """Freeze/seal a subscriber and request writer-joined lag recovery.

        Must be called under ``_global_lock``.  If a replayable transport write
        is in flight, field capture is deliberately deferred until that write
        reports success/failure.  This is the join-before-capture boundary.
        """
        if sub.closed:
            return
        sub.closed = True
        sub.lag_pending = True
        # The triggering event is abandoned, as is the sealed queue.  Do not
        # capture the cursor while a write can still change ground truth.
        # Settle-release all queue charges.
        queue_settled = self._sub_queue_settled.pop((run.run_id, sub_id), 0)
        if queue_settled > 0:
            self._budget.settle_release("subscriber_queue", queue_settled)
            self._sub_queue_settled[(run.run_id, sub_id)] = 0
        sub.queue.clear()
        sub.queue_event_count = 0
        sub.queue_byte_count = 0
        if not sub.slot_acquired:
            self._finalize_lag_locked(run, sub)

    def _finalize_lag_locked(self, run: _RunState, sub: _Subscriber) -> None:
        """Capture lag fields after transport quiescence (lock already held)."""
        if not sub.lag_pending or sub.slot_acquired:
            return
        last_delivered = sub.last_delivered_event_id
        latest_available = run.ring.max_event_id
        lag_data = {
            "event": "hermes.run_events.subscriber.lagged",
            "last_delivered_event_id": last_delivered,
            "first_dropped_event_id": last_delivered + 1,
            "latest_available_event_id": latest_available,
            "dropped_events": latest_available - last_delivered,
        }
        lag_bytes = self._serialize_event("hermes.run_events.subscriber.lagged", lag_data)
        lag_frame = self._format_control_frame(
            "hermes.run_events.subscriber.lagged", lag_bytes
        )
        # Settle the control frame charge (Phase 2, within reserved control_frame budget).
        lag_frame_size = len(lag_frame)
        self._budget.settle("control_frame", lag_frame_size)
        sub.pending_lag_frame = lag_frame
        sub.pending_lag_frame_size = lag_frame_size
        sub.lag_pending = False

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
        """Get the next SSE frame for a subscriber.

        Acquires the serialization slot (FREE→ACQUIRED) and returns the
        formatted frame **without** advancing ``last_delivered_event_id``.
        The caller MUST call :meth:`confirm_delivered` after a successful
        transport write, or :meth:`mark_write_failed` if the write fails.

        Returns formatted bytes, or None if the subscriber is closed and
        has no pending lag frame.
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            sub = run.subscribers.get(sub_id)
            if sub is None:
                return None
            # If there's a pending lag frame, deliver it and signal close.
            # Lag frames are control frames — no slot acquisition needed.
            if sub.pending_lag_frame is not None:
                frame = sub.pending_lag_frame
                sub.pending_lag_frame = None
                return frame
            if sub.closed:
                return None
            # Exactly one serialized copy may be outstanding.  The sole writer
            # awaits completion; no second dequeue may overwrite slot state.
            if sub.slot_acquired:
                return None
            # Drain the queue — acquire serialization slot.
            if sub.queue:
                event_id, event_name, data_bytes, frame_size = sub.queue.popleft()
                sub.queue_event_count -= 1
                sub.queue_byte_count -= frame_size
                # Settle-release the queue charge (item dequeued for delivery).
                self._budget.settle_release("subscriber_queue", frame_size)
                self._sub_queue_settled[(run_id, sub_id)] = max(
                    0, self._sub_queue_settled.get((run_id, sub_id), 0) - frame_size
                )
                # Acquire slot (FREE → ACQUIRED) and settle serialization copy.
                sub.slot_acquired = True
                sub.slot_acquired_at = time.monotonic()
                sub.in_flight_event_id = event_id
                sub.in_flight_event_name = event_name
                sub.in_flight_frame_size = frame_size
                # NOTE: last_delivered_event_id is NOT advanced here.
                # confirm_delivered() advances it after transport success.
                frame = self._format_replayable_frame(event_name, event_id, data_bytes)
                # Phase 2: settle the serialized copy charge.
                self._budget.settle("serialization", frame_size)
                return frame
            return None  # no data right now

    def acquire_replay_frame(
        self, run_id: str, sub_id: int, entry: _RingEntry
    ) -> Optional[bytes]:
        """Acquire the subscriber's one serialization slot for replay."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            sub = run.subscribers.get(sub_id)
            if sub is None or sub.closed or sub.slot_acquired:
                return None
            sub.slot_acquired = True
            sub.slot_acquired_at = time.monotonic()
            sub.in_flight_event_id = entry.event_id
            sub.in_flight_event_name = entry.event_name
            frame_size = self._frame_size(entry.event_name, entry.data, entry.event_id)
            sub.in_flight_frame_size = frame_size
            frame = self._format_replayable_frame(
                entry.event_name, entry.event_id, entry.data
            )
            # Phase 2: settle the serialized copy charge.
            self._budget.settle("serialization", frame_size)
            return frame

    def confirm_delivered(self, run_id: str, sub_id: int) -> None:
        """Confirm that the in-flight frame was delivered to the transport.

        Advances ``last_delivered_event_id`` to the in-flight event's ID
        and releases the serialization slot (ACQUIRED → FREE).

        No-op if no slot is acquired (e.g., lag/control frame was delivered).
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            sub = run.subscribers.get(sub_id)
            if sub is None:
                return
            if sub.slot_acquired:
                # A frozen subscriber still advances when the joined in-flight
                # write succeeded; capture happens only after this transition.
                sub.last_delivered_event_id = sub.in_flight_event_id
                # Settle-release the serialization copy charge.
                if sub.in_flight_frame_size > 0:
                    self._budget.settle_release("serialization", sub.in_flight_frame_size)
                sub.slot_acquired = False
                sub.in_flight_event_id = 0
                sub.in_flight_event_name = ""
                sub.in_flight_frame_size = 0
                self._finalize_lag_locked(run, sub)

    def mark_write_failed(self, run_id: str, sub_id: int) -> None:
        """Release the serialization slot without advancing last_delivered.

        Called when the transport write failed.  The in-flight event is
        NOT counted as delivered.
        """
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            sub = run.subscribers.get(sub_id)
            if sub is None:
                return
            if sub.slot_acquired:
                # Settle-release the serialization copy charge.
                if sub.in_flight_frame_size > 0:
                    self._budget.settle_release("serialization", sub.in_flight_frame_size)
                sub.slot_acquired = False
                sub.in_flight_event_id = 0
                sub.in_flight_event_name = ""
                sub.in_flight_frame_size = 0
                self._finalize_lag_locked(run, sub)

    def check_serialization_timeout(
        self, run_id: str, sub_id: int, now: Optional[float] = None
    ) -> bool:
        """Check if the subscriber's serialization slot has been held too long.

        Per §2 (serialization-timeout transition), this entry fires only when
        the slot could not be acquired for a **next replayable event for this
        subscriber** — specifically, when the subscriber's own queue is
        non-empty (there is a real pending event this subscriber would
        receive).  A filtered-out later global event that never enters this
        subscriber's queue must NOT trigger timeout.

        The pending event (queue head) is the triggering event — a real
        extant event preserved through join→capture as the abandoned-interval
        head.  If the queue is empty, there is no pending event and the
        timeout must not fire (per-subscriber backpressure, not lag).
        """
        if now is None:
            now = time.monotonic()
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            sub = run.subscribers.get(sub_id)
            if sub is None or sub.closed:
                return False
            # Only fire if:
            # 1. The slot has been held longer than one heartbeat.
            # 2. There is a real pending next event FOR THIS SUBSCRIBER
            #    in the queue (the subscriber-local pending candidate).
            # A filtered-out later global event that never enters this
            # subscriber's queue does not satisfy condition 2.
            if (
                sub.slot_acquired
                and len(sub.queue) > 0
                and (now - sub.slot_acquired_at) > self.snapshot.heartbeat_seconds
            ):
                # Serialization-timeout → close-on-lag (§2, §6).
                # The triggering event is the subscriber's queue head —
                # the real pending event that was waiting for slot
                # acquisition.
                trigger_id, trigger_name, _, trigger_frame_size = sub.queue[0]
                self._close_on_lag(
                    run, sub_id, sub,
                    triggering_event_id=trigger_id,
                    triggering_event_name=trigger_name,
                    triggering_frame_size=trigger_frame_size,
                )
                return True
            return False

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

    def get_final_event_id(self, run_id: str) -> int:
        """Return the ring's max replayable event ID (for terminal frames)."""
        with self._global_lock:
            run = self._runs.get(run_id)
            if run is None:
                return 0
            return run.ring.max_event_id

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
    def budget_observability(self) -> Dict[str, int]:
        """All 24 dual-ledger counters (§2 test interface).

        Returns the full set of current and high-water counters for all
        five budget classes and the total.
        """
        with self._global_lock:
            return self._budget.observability_snapshot()

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
        """Observable container-budget charged-byte counter (settled-current)."""
        with self._global_lock:
            return self._budget.classes["container"].current_charged_bytes

    @property
    def container_high_water(self) -> int:
        """High-water mark of container budget usage (settled-current)."""
        with self._global_lock:
            return self._budget.classes["container"].current_high_water

    def charge_container(self, nbytes: int) -> None:
        """Charge *nbytes* to the container budget (diagnostic/testing helper).

        Uses the dual-ledger atomic admission path (reserve + settle).
        """
        with self._global_lock:
            self._budget.reserve_and_settle_container(
                nbytes,
                self.snapshot.container_budget_bytes,
                self.snapshot.total_feature_memory_budget_bytes,
            )

    def release_container(self, nbytes: int) -> None:
        """Release *nbytes* from the container budget."""
        with self._global_lock:
            self._budget.release_container(nbytes)

    @property
    def ring_high_water_bytes(self) -> Dict[str, int]:
        """Observable ring byte high-water per run.

        Returns a dict of run_id → high_water_bytes.  Each ring's
        high-water must never exceed ``max_replay_bytes``.
        """
        with self._global_lock:
            return {rid: r.ring.high_water_bytes for rid, r in self._runs.items()}
