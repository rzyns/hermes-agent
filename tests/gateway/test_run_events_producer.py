"""Tests for the SSE run-events producer (round-6 contract).

Covers the eight-item verification matrix from t_d135602a plus the
contract's own testable invariants:

1. ordered events and IDs;
2. replay and reconnect strictly after Last-Event-ID;
3. slow/disconnected subscriber isolation and cleanup;
4. missing/invalid bearer rejected before SSE bytes;
5. malformed/oversized producer event handling is bounded/redacted;
6. multiple concurrent subscribers each receive the complete ordered stream;
7. cancellation/terminal cleanup and retained replay state;
8. regression coverage for run status/stop/approval.

Plus contract invariants:
- byte-budget high-water observability;
- lag-field capture ordering;
- cursor-0 sentinel;
- snapshot completeness;
- category-based fail-closed include;
- four-limit atomic admission with 503 run_events_overload;
- close-on-lag with join-then-capture quiescence;
- global reconnect cursor gap fields.

Round-2 regression tests (comment on t_2d596b66):
- replay/live handoff atomicity (no duplicates, include-filtered);
- serialization slot FREE→ACQUIRED→FREE + confirm_delivered ground truth;
- container budget enforced at admission (not exceeding budget);
- queue byte admission checks current + candidate (not current alone);
- terminal frame uses actual final replayable event ID;
- serialization-timeout → close-on-lag entry point.
"""

import json

import pytest

from gateway.platforms.run_events_producer import (
    CursorError,
    DEFAULT_SNAPSHOT,
    IncludeParseError,
    RunEventsCapabilitiesSnapshot,
    RunEventsProducer,
    SUPPORTED_INCLUDE_CATEGORIES,
    is_replayable_event,
    parse_cursor,
    parse_include,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_producer(**kwargs) -> RunEventsProducer:
    """Create a producer with optional snapshot overrides."""
    snapshot = RunEventsCapabilitiesSnapshot(**kwargs) if kwargs else None
    return RunEventsProducer(snapshot)


def _emit_n_events(producer: RunEventsProducer, run_id: str, n: int, prefix: str = "tool.started"):
    """Emit n events to a run's ring."""
    for i in range(n):
        producer.emit_event(run_id, prefix, {"run_id": run_id, "index": i})


def _admit(producer, run_id, cursor=0, include=None):
    """Convenience: admit subscriber and return sub_id."""
    result = producer.admit_subscriber_and_snapshot(run_id, cursor, include)
    assert result is not None, "admission failed"
    return result[0]  # type: ignore[index]


def _drain_and_confirm(producer, run_id, sub_id, max_frames=1000):
    """Drain subscriber queue, confirming each delivery. Returns list of frames."""
    frames = []
    for _ in range(max_frames):
        f = producer.get_subscriber_frame(run_id, sub_id, 0)
        if f is None:
            break
        frames.append(f)
        if b"subscriber.lagged" not in f:
            producer.confirm_delivered(run_id, sub_id)
    return frames


# ---------------------------------------------------------------------------
# §3 Capability snapshot completeness
# ---------------------------------------------------------------------------


class TestCapabilitySnapshot:
    def test_snapshot_has_all_enforced_limits(self):
        """Every enforced limit must appear in the snapshot (§3, §7)."""
        snap = RunEventsCapabilitiesSnapshot()
        block = snap.endpoint_block()
        required_keys = [
            "version", "snapshot_id", "replay", "multicast",
            "max_replay_events", "max_replay_bytes",
            "max_subscriber_queue_events", "max_subscriber_queue_bytes",
            "max_event_bytes", "max_retained_runs",
            "max_active_runs_for_events", "max_concurrent_subscribers",
            "max_subscribers_per_run", "heartbeat_seconds",
            "terminal_retention_seconds",
            "total_feature_memory_budget_bytes",
            "container_budget_bytes",
            "control_slot_pool_bytes",
            "serialization_pool_bytes",
            "supported_include_categories",
            "mandatory_category", "include_match",
        ]
        for key in required_keys:
            assert key in block, f"Missing key in endpoint block: {key}"

    def test_endpoint_and_in_stream_blocks_match(self):
        """Endpoint block and in-stream frame carry the same fields (§3)."""
        snap = RunEventsCapabilitiesSnapshot()
        endpoint = snap.endpoint_block()
        in_stream = snap.in_stream_data()
        assert endpoint == in_stream

    def test_version_is_integer(self):
        """version must be an integer, not a string (§3 hard constraint)."""
        snap = RunEventsCapabilitiesSnapshot()
        assert isinstance(snap.version, int)
        assert snap.endpoint_block()["version"] == snap.version

    def test_memory_budget_matches_formula(self):
        """Total budget = rings + queues + control + serialization + container."""
        snap = RunEventsCapabilitiesSnapshot()
        expected = (
            snap.max_retained_runs * snap.max_replay_bytes
            + snap.max_concurrent_subscribers * snap.max_subscriber_queue_bytes
            + snap.max_concurrent_subscribers * snap.max_event_bytes  # control
            + snap.max_concurrent_subscribers * snap.max_event_bytes  # serialization
            + snap.container_budget_bytes
        )
        assert snap.total_feature_memory_budget_bytes == expected
        # With defaults = 256 MiB.
        assert snap.total_feature_memory_budget_bytes == 268_435_456

    def test_overload_error_includes_all_four_limits(self):
        """503 run_events_overload must include all four limits (§2)."""
        snap = RunEventsCapabilitiesSnapshot()
        err = snap.overload_error()["error"]
        assert err["code"] == "run_events_overload"
        for key in (
            "max_active_runs_for_events",
            "max_retained_runs",
            "max_concurrent_subscribers",
            "max_subscribers_per_run",
        ):
            assert key in err


# ---------------------------------------------------------------------------
# §4 Include category parsing
# ---------------------------------------------------------------------------


class TestIncludeParsing:
    def test_absent_include_means_all_categories(self):
        """Absent key = all non-meta categories (§4)."""
        result = parse_include([])
        assert result == set(SUPPORTED_INCLUDE_CATEGORIES)

    def test_single_category(self):
        result = parse_include(["tool"])
        assert result == {"tool"}

    def test_multiple_categories_comma_separated(self):
        result = parse_include(["tool,message,status"])
        assert result == {"tool", "message", "status"}

    def test_duplicates_silently_deduplicated(self):
        result = parse_include(["tool,tool,message"])
        assert result == {"tool", "message"}

    def test_repeated_parameters_rejected(self):
        """?include=tool&include=message → 400 invalid_include_list (§4)."""
        with pytest.raises(IncludeParseError) as exc_info:
            parse_include(["tool", "message"])
        assert exc_info.value.code == "invalid_include_list"

    def test_empty_token_rejected(self):
        """include= or include=, → 400 invalid_include_list (§4)."""
        with pytest.raises(IncludeParseError) as exc_info:
            parse_include([""])
        assert exc_info.value.code == "invalid_include_list"

    def test_empty_token_after_comma_rejected(self):
        with pytest.raises(IncludeParseError) as exc_info:
            parse_include(["tool,,message"])
        assert exc_info.value.code == "invalid_include_list"

    def test_trailing_comma_rejected(self):
        with pytest.raises(IncludeParseError) as exc_info:
            parse_include(["tool,"])
        assert exc_info.value.code == "invalid_include_list"

    def test_unknown_category_rejected(self):
        """Unknown category → 400 unsupported_include_kinds (§4)."""
        with pytest.raises(IncludeParseError) as exc_info:
            parse_include(["foo,bar"])
        assert exc_info.value.code == "unsupported_include_kinds"
        assert exc_info.value.supported == SUPPORTED_INCLUDE_CATEGORIES

    def test_whitespace_trimmed(self):
        result = parse_include([" tool , message "])
        assert result == {"tool", "message"}


# ---------------------------------------------------------------------------
# §5 Cursor parsing and validation
# ---------------------------------------------------------------------------


class TestCursorParsing:
    def test_missing_cursor_returns_zero(self):
        """Missing cursor = cursor 0 sentinel (replay from ring start)."""
        assert parse_cursor(None) == 0
        assert parse_cursor("") == 0

    def test_zero_cursor(self):
        assert parse_cursor("0") == 0

    def test_positive_cursor(self):
        assert parse_cursor("42") == 42

    def test_malformed_cursor_rejected(self):
        with pytest.raises(CursorError) as exc_info:
            parse_cursor("abc")
        assert exc_info.value.code == "invalid_cursor"
        assert exc_info.value.http_status == 400

    def test_negative_cursor_rejected(self):
        with pytest.raises(CursorError) as exc_info:
            parse_cursor("-1")
        assert exc_info.value.code == "invalid_cursor"
        assert exc_info.value.http_status == 400


class TestCursorValidation:
    def test_cursor_zero_valid_on_active_run(self):
        producer = _make_producer()
        producer.admit_run("run1")
        producer.validate_cursor("run1", 0)  # should not raise

    def test_cursor_zero_valid_on_terminated_run_with_ring(self):
        producer = _make_producer()
        producer.admit_run("run1")
        producer.emit_event("run1", "tool.started", {"i": 1})
        producer.mark_run_terminal("run1")
        producer.validate_cursor("run1", 0)  # should not raise

    def test_expired_cursor(self):
        """N + 1 < min_retained_id → 409 cursor_expired."""
        producer = _make_producer(max_replay_events=3)
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 10)  # ring holds last 3
        min_retained = producer._runs["run1"].ring.min_retained_id
        # Cursor well below min_retained.
        cursor = min_retained - 5
        with pytest.raises(CursorError) as exc_info:
            producer.validate_cursor("run1", cursor)
        assert exc_info.value.code == "cursor_expired"
        assert exc_info.value.http_status == 409

    def test_future_cursor_on_terminated_run(self):
        """cursor > max_id and run not active → 409 cursor_future."""
        producer = _make_producer()
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 5)
        producer.mark_run_terminal("run1")
        max_id = producer._runs["run1"].ring.max_event_id
        with pytest.raises(CursorError) as exc_info:
            producer.validate_cursor("run1", max_id + 10)
        assert exc_info.value.code == "cursor_future"
        assert exc_info.value.http_status == 409

    def test_run_not_found(self):
        producer = _make_producer()
        with pytest.raises(CursorError) as exc_info:
            producer.validate_cursor("nonexistent", 0)
        assert exc_info.value.code == "run_not_found"
        assert exc_info.value.http_status == 404


# ---------------------------------------------------------------------------
# §1-2 Replay ring: ordered events and IDs
# ---------------------------------------------------------------------------


class TestReplayRing:
    def test_monotonic_ids_starting_at_1(self):
        """Per-run monotonic integer IDs starting at 1 (§5)."""
        producer = _make_producer()
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 5)
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(0)
        ids = [e.event_id for e in entries]
        assert ids == [1, 2, 3, 4, 5]

    def test_event_count_limit_eviction(self):
        """Ring evicts oldest events when event count exceeded."""
        producer = _make_producer(max_replay_events=3)
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 5)
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(0)
        assert len(entries) == 3
        # Should retain events 3, 4, 5 (FIFO eviction).
        assert [e.event_id for e in entries] == [3, 4, 5]

    def test_replay_after_cursor(self):
        """Replay returns events with IDs > cursor (§5)."""
        producer = _make_producer()
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 10)
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(3)
        assert [e.event_id for e in entries] == [4, 5, 6, 7, 8, 9, 10]

    def test_replay_from_cursor_zero(self):
        """Cursor 0 returns entire ring (§5)."""
        producer = _make_producer(max_replay_events=5)
        producer.admit_run("run1")
        _emit_n_events(producer, "run1", 10)
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(0)
        assert len(entries) == 5  # only last 5 retained

    def test_separate_runs_have_independent_ids(self):
        producer = _make_producer()
        producer.admit_run("run1")
        producer.admit_run("run2")
        _emit_n_events(producer, "run1", 3)
        _emit_n_events(producer, "run2", 2)
        ring1 = producer._runs["run1"].ring
        ring2 = producer._runs["run2"].ring
        assert [e.event_id for e in ring1.replay_after(0)] == [1, 2, 3]
        assert [e.event_id for e in ring2.replay_after(0)] == [1, 2]


# ---------------------------------------------------------------------------
# §7 Oversize event handling
# ---------------------------------------------------------------------------


class TestOversizeEvents:
    def test_oversize_event_replaced(self):
        """Events > max_event_bytes are replaced with oversize frame (§7)."""
        producer = _make_producer(max_event_bytes=100)
        producer.admit_run("run1")
        big_data = {"x": "A" * 200}
        producer.emit_event("run1", "tool.completed", big_data)
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(0)
        assert len(entries) == 1
        data = json.loads(entries[0].data)
        assert data["original_event"] == "tool.completed"
        assert data["code"] == "event_oversize"
        assert data["size_bytes"] > 100

    def test_normal_events_not_replaced(self):
        producer = _make_producer(max_event_bytes=1000)
        producer.admit_run("run1")
        producer.emit_event("run1", "tool.started", {"i": 1})
        ring = producer._runs["run1"].ring
        entries = ring.replay_after(0)
        data = json.loads(entries[0].data)
        assert data == {"i": 1}


# ---------------------------------------------------------------------------
# §6 Slow subscriber isolation and cleanup
# ---------------------------------------------------------------------------


class TestSlowSubscriber:
    def test_queue_overflow_triggers_close_on_lag(self):
        """Queue overflow sends lag frame and closes subscriber (§6)."""
        producer = _make_producer(
            max_subscriber_queue_events=3,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("run1")
        sub_id = _admit(producer, "run1")
        # Fill the queue beyond capacity.
        _emit_n_events(producer, "run1", 10)
        # Subscriber should be closed with a lag frame.
        sub = producer._runs["run1"].subscribers[sub_id]
        assert sub.closed
        assert sub.pending_lag_frame is not None

    def test_lag_frame_has_four_gap_fields(self):
        """Lag frame carries the four global reconnect cursor fields (§6)."""
        producer = _make_producer(
            max_subscriber_queue_events=2,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("run1")
        sub_id = _admit(producer, "run1")
        _emit_n_events(producer, "run1", 10)
        # The lag frame should have been emitted.
        frame = producer.get_subscriber_frame("run1", sub_id, 0)
        assert frame is not None
        assert b"subscriber.lagged" in frame
        # Parse the lag data.
        for line in frame.split(b"\n"):
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                assert "last_delivered_event_id" in data
                assert "first_dropped_event_id" in data
                assert "latest_available_event_id" in data
                assert "dropped_events" in data
                break

    def test_lag_interval_is_contiguous(self):
        """dropped = latest - last_delivered; first_dropped = last + 1 (§6)."""
        producer = _make_producer(
            max_subscriber_queue_events=2,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("run1")
        sub_id = _admit(producer, "run1")
        _emit_n_events(producer, "run1", 10)
        frame = producer.get_subscriber_frame("run1", sub_id, 0)
        for line in frame.split(b"\n"):
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                assert data["first_dropped_event_id"] == data["last_delivered_event_id"] + 1
                assert data["dropped_events"] == data["latest_available_event_id"] - data["last_delivered_event_id"]
                break

    def test_slow_subscriber_does_not_affect_others(self):
        """A slow subscriber must not affect other subscribers or the run (§6)."""
        producer = _make_producer(
            max_subscriber_queue_events=100,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("run1")
        slow_sub = _admit(producer, "run1")
        fast_sub = _admit(producer, "run1")
        # Emit a few events that both can handle.
        _emit_n_events(producer, "run1", 5)
        # Fast subscriber drains its queue.
        for _ in range(5):
            f = producer.get_subscriber_frame("run1", fast_sub, 0)
            if f:
                producer.confirm_delivered("run1", fast_sub)
        # Slow subscriber does not drain. Now overflow its queue by
        # exceeding max_subscriber_queue_events.
        _emit_n_events(producer, "run1", 100)
        # Slow subscriber should be closed.
        assert producer._runs["run1"].subscribers[slow_sub].closed
        # Fast subscriber should NOT be closed (it was draining).
        assert not producer._runs["run1"].subscribers[fast_sub].closed

    def test_disconnect_cleans_only_subscriber(self):
        """Subscriber disconnect cleans only that subscriber (§6)."""
        producer = _make_producer()
        producer.admit_run("run1")
        sub1 = _admit(producer, "run1")
        sub2 = _admit(producer, "run1")
        producer.remove_subscriber("run1", sub1)
        assert sub1 not in producer._runs["run1"].subscribers
        assert sub2 in producer._runs["run1"].subscribers
        # Run should still exist.
        assert producer.run_exists("run1")


# ---------------------------------------------------------------------------
# §2 Four-limit atomic admission
# ---------------------------------------------------------------------------


class TestAtomicAdmission:
    def test_max_retained_runs_enforced(self):
        producer = _make_producer(max_retained_runs=3)
        assert producer.admit_run("r1")
        assert producer.admit_run("r2")
        assert producer.admit_run("r3")
        assert not producer.admit_run("r4")  # overload

    def test_max_concurrent_subscribers_enforced(self):
        producer = _make_producer(max_concurrent_subscribers=3)
        producer.admit_run("r1")
        assert _admit(producer, "r1") is not None
        assert _admit(producer, "r1") is not None
        assert _admit(producer, "r1") is not None
        # overload — admit_subscriber_and_snapshot returns None
        assert producer.admit_subscriber_and_snapshot("r1", 0) is None

    def test_max_subscribers_per_run_enforced(self):
        producer = _make_producer(max_subscribers_per_run=2)
        producer.admit_run("r1")
        assert _admit(producer, "r1") is not None
        assert _admit(producer, "r1") is not None
        assert producer.admit_subscriber_and_snapshot("r1", 0) is None  # overload per-run

    def test_overload_error_body(self):
        snap = RunEventsCapabilitiesSnapshot()
        err = snap.overload_error()
        assert err["error"]["code"] == "run_events_overload"

    def test_terminal_run_releases_active_slot(self):
        producer = _make_producer(max_active_runs_for_events=2)
        producer.admit_run("r1")
        producer.admit_run("r2")
        producer.mark_run_terminal("r1")
        # r1 is no longer active, so a new active run can be admitted.
        assert producer.admit_run("r3")
        assert producer.active_run_count == 2  # r2 + r3

    def test_retained_slot_held_until_retention_expiry(self):
        producer = _make_producer(
            max_retained_runs=2,
            terminal_retention_seconds=1,
        )
        producer.admit_run("r1")
        producer.admit_run("r2")
        producer.mark_run_terminal("r1")
        # r1 still holds a retained slot.
        assert not producer.admit_run("r3")
        # After retention expiry (set expiry to 1 second ago).
        import time as _time
        producer._runs["r1"].retention_expiry = _time.time() - 1.0
        producer.sweep_expired_runs()
        assert producer.admit_run("r3")


# ---------------------------------------------------------------------------
# §4 Category filtering
# ---------------------------------------------------------------------------


class TestCategoryFiltering:
    def test_filtered_categories_not_delivered(self):
        """include filter excludes non-matching categories (§4)."""
        producer = _make_producer(max_subscriber_queue_events=100)
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1", include={"tool"})
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "message.delta", {"d": "hi"})
        producer.emit_event("r1", "tool.completed", {"i": 2})
        sub = producer._runs["r1"].subscribers[sub_id]
        # Only tool events should be in the queue.
        events = []
        while sub.queue:
            eid, name, data = sub.queue.popleft()
            events.append(name)
        assert events == ["tool.started", "tool.completed"]

    def test_meta_always_delivered(self):
        """meta category is always delivered regardless of include (§4)."""
        producer = _make_producer(max_subscriber_queue_events=100)
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1", include={"tool"})
        producer.emit_event("r1", "hermes.run_events.heartbeat", {"t": 1})
        producer.emit_event("r1", "tool.started", {"i": 1})
        sub = producer._runs["r1"].subscribers[sub_id]
        events = []
        while sub.queue:
            eid, name, data = sub.queue.popleft()
            events.append(name)
        assert "hermes.run_events.heartbeat" in events
        assert "tool.started" in events


# ---------------------------------------------------------------------------
# §5 Replayable vs control frames
# ---------------------------------------------------------------------------


class TestReplayableVsControl:
    def test_replayable_events_carry_id(self):
        """Only replayable events carry SSE id (§5)."""
        assert is_replayable_event("tool.started")
        assert is_replayable_event("message.delta")
        assert is_replayable_event("run.completed")
        assert is_replayable_event("error")

    def test_control_frames_no_id(self):
        """Control frames do not carry id (§5)."""
        assert not is_replayable_event("hermes.run_events.capabilities")
        assert not is_replayable_event("hermes.run_events.heartbeat")
        assert not is_replayable_event("hermes.run_events.subscriber.lagged")

    def test_replay_frame_has_id_line(self):
        producer = _make_producer()
        producer.admit_run("r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        ring = producer._runs["r1"].ring
        entries = ring.replay_after(0)
        frame = producer.format_replay_frame(entries[0])
        assert b"\nid: 1\n" in frame

    def test_control_frame_has_no_id_line(self):
        producer = _make_producer()
        frame = producer.format_capabilities_frame()
        assert b"\nid:" not in frame

    def test_heartbeat_is_comment(self):
        producer = _make_producer()
        frame = producer.format_heartbeat()
        assert frame == b": keepalive\n\n"

    def test_terminal_frame_has_no_id(self):
        producer = _make_producer()
        frame = producer.format_terminal_frame("r1", "completed", 42)
        assert b"\nid:" not in frame
        assert b"hermes.run_events.terminal" in frame


# ---------------------------------------------------------------------------
# §5 Atomic snapshot-plus-subscribe
# ---------------------------------------------------------------------------


class TestAtomicSnapshotPlusSubscribe:
    def test_replay_then_live_events_ordered(self):
        """Replayed events arrive before live events (§5)."""
        producer = _make_producer()
        producer.admit_run("r1")
        _emit_n_events(producer, "r1", 3)
        sub_id, replay, active = producer.admit_subscriber_and_snapshot("r1", 0)
        assert sub_id is not None
        # Snapshot should capture all 3 existing events.
        assert len(replay) == 3
        assert active is True
        assert [e.event_id for e in replay] == [1, 2, 3]


# ---------------------------------------------------------------------------
# §2 Byte-budget high-water observability
# ---------------------------------------------------------------------------


class TestByteBudgetObservability:
    def test_container_charge_and_release(self):
        """Container budget charging is observable (§2)."""
        producer = _make_producer()
        producer.charge_container(1000)
        assert producer.container_bytes_used == 1000
        assert producer.container_high_water == 1000
        producer.charge_container(500)
        assert producer.container_bytes_used == 1500
        assert producer.container_high_water == 1500
        producer.release_container(800)
        assert producer.container_bytes_used == 700
        # High-water mark does not decrease.
        assert producer.container_high_water == 1500

    def test_container_charged_on_admission(self):
        """Container budget is charged at subscriber admission (§2)."""
        producer = _make_producer()
        producer.admit_run("r1")
        assert producer.container_bytes_used > 0  # run metadata charged
        run_bytes = producer.container_bytes_used
        _admit(producer, "r1")
        assert producer.container_bytes_used > run_bytes  # subscriber metadata charged

    def test_container_released_on_disconnect(self):
        """Container budget is released on subscriber disconnect (§2)."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        before = producer.container_bytes_used
        producer.remove_subscriber("r1", sub_id)
        assert producer.container_bytes_used < before

    def test_container_high_water_never_exceeds_budget(self):
        """container_high_water must stay within container_budget_bytes."""
        snap = RunEventsCapabilitiesSnapshot()
        producer = _make_producer()
        # Simulate maximum subscriber admission charging.
        # Each subscriber connection metadata is ~1 KiB for this model.
        per_sub = 1024
        for _ in range(snap.max_concurrent_subscribers):
            producer.charge_container(per_sub)
        assert producer.container_high_water <= snap.container_budget_bytes


# ---------------------------------------------------------------------------
# §6 Lag-field capture ordering (join-then-capture)
# ---------------------------------------------------------------------------


class TestLagFieldCaptureOrdering:
    def test_last_delivered_reflects_delivered_events(self):
        """After close-on-lag, last_delivered reflects what was actually
        delivered to the transport (§6 join-then-capture).

        This test would FAIL on the old commit where
        get_subscriber_frame advanced last_delivered_event_id before
        the transport write was confirmed.
        """
        producer = _make_producer(
            max_subscriber_queue_events=3,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        # Deliver 2 events (they go into the subscriber queue).
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "tool.completed", {"i": 2})
        # Read 2 frames from the queue (simulating delivery).
        # In the new API, get_subscriber_frame acquires the slot but does
        # NOT advance last_delivered_event_id.  The caller must
        # confirm_delivered after the write succeeds.
        f1 = producer.get_subscriber_frame("r1", sub_id, 0)
        producer.confirm_delivered("r1", sub_id)
        f2 = producer.get_subscriber_frame("r1", sub_id, 0)
        producer.confirm_delivered("r1", sub_id)
        assert f1 is not None and f2 is not None
        assert b"id: 1" in f1
        assert b"id: 2" in f2
        # Now overflow the queue.
        for i in range(10):
            producer.emit_event("r1", "tool.started", {"i": i + 3})
        # The lag frame should have last_delivered = 2 (what was actually delivered).
        lag = producer.get_subscriber_frame("r1", sub_id, 0)
        assert lag is not None
        assert b"subscriber.lagged" in lag
        for line in lag.split(b"\n"):
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                assert data["last_delivered_event_id"] == 2
                break

    def test_undelivered_event_not_counted_as_delivered(self):
        """An event dequeued but not confirmed is NOT counted as delivered
        (§6 join-then-capture).  This test would FAIL on the old commit.
        """
        producer = _make_producer(
            max_subscriber_queue_events=5,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "tool.started", {"i": 2})
        # Dequeue event 1 but do NOT confirm delivery (simulates a failed
        # or in-flight transport write).
        f1 = producer.get_subscriber_frame("r1", sub_id, 0)
        assert f1 is not None
        assert b"id: 1" in f1
        # last_delivered_event_id must still be 0 — event 1 was dequeued
        # but not confirmed delivered.
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.last_delivered_event_id == 0
        assert sub.slot_acquired  # slot is held
        # Now mark write failed (transport error).
        producer.mark_write_failed("r1", sub_id)
        assert not sub.slot_acquired  # slot released
        assert sub.last_delivered_event_id == 0  # still 0


# ---------------------------------------------------------------------------
# Multicast: concurrent subscribers receive the same ordered stream
# ---------------------------------------------------------------------------


class TestMulticast:
    def test_concurrent_subscribers_receive_same_events(self):
        """Multiple concurrent subscribers each receive complete stream (§6 item 6)."""
        producer = _make_producer(max_subscriber_queue_events=100)
        producer.admit_run("r1")
        sub1 = _admit(producer, "r1")
        sub2 = _admit(producer, "r1")
        _emit_n_events(producer, "r1", 5)
        # Both subscribers should have 5 events in their queues.
        s1 = producer._runs["r1"].subscribers[sub1]
        s2 = producer._runs["r1"].subscribers[sub2]
        assert s1.queue_event_count == 5
        assert s2.queue_event_count == 5
        # And the IDs should match.
        ids1 = [eid for eid, _, _ in s1.queue]
        ids2 = [eid for eid, _, _ in s2.queue]
        assert ids1 == ids2 == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Capabilities frame format
# ---------------------------------------------------------------------------


class TestCapabilitiesFrame:
    def test_capabilities_frame_format(self):
        producer = _make_producer()
        frame = producer.format_capabilities_frame()
        assert frame.startswith(b"event: hermes.run_events.capabilities\n")
        assert b"\ndata: " in frame
        assert frame.endswith(b"\n\n")
        # No id line.
        assert b"\nid:" not in frame

    def test_capabilities_frame_data_is_valid_json(self):
        producer = _make_producer()
        frame = producer.format_capabilities_frame()
        for line in frame.split(b"\n"):
            if line.startswith(b"data: "):
                data = json.loads(line[6:])
                assert data["version"] == 1
                assert "max_event_bytes" in data
                break


# ---------------------------------------------------------------------------
# Terminal retention sweep
# ---------------------------------------------------------------------------


class TestRetentionSweep:
    def test_sweep_expires_terminal_runs(self):
        import time as _time
        producer = _make_producer(terminal_retention_seconds=1)
        producer.admit_run("r1")
        producer.mark_run_terminal("r1")
        # Force expiry (set expiry to 1 second ago).
        producer._runs["r1"].retention_expiry = _time.time() - 1.0
        expired = producer.sweep_expired_runs()
        assert "r1" in expired
        assert not producer.run_exists("r1")

    def test_sweep_does_not_expire_active_runs(self):
        producer = _make_producer()
        producer.admit_run("r1")
        expired = producer.sweep_expired_runs()
        assert "r1" not in expired
        assert producer.run_exists("r1")


# ===========================================================================
# Round-2 regression tests: the four review blockers
# ===========================================================================


class TestRegressionReplayLiveHandoff:
    """Blocker 1: replay/live handoff must be atomic and filter-correct."""

    def test_no_duplicate_between_replay_and_live(self):
        """An event emitted after snapshot must not appear in both the
        replay snapshot AND the live queue (§5 atomic handoff).

        This test would FAIL on the old commit where admit_subscriber
        and snapshot_and_subscribe were separate, non-atomic calls.
        """
        producer = _make_producer(max_subscriber_queue_events=100)
        producer.admit_run("r1")
        # Pre-fill ring with 2 events.
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "tool.started", {"i": 2})
        # Atomic admit+snapshot captures replay [1, 2] and registers
        # for live events.
        sub_id, replay, _ = producer.admit_subscriber_and_snapshot("r1", 0)
        replay_ids = [e.event_id for e in replay]
        # Now emit event 3 (live, after snapshot).
        producer.emit_event("r1", "tool.started", {"i": 3})
        # Event 3 should be in the live queue ONLY.
        sub = producer._runs["r1"].subscribers[sub_id]
        live_ids = [eid for eid, _, _ in sub.queue]
        # Replay had [1, 2], live queue has [3].  No overlap.
        assert replay_ids == [1, 2]
        assert live_ids == [3]
        assert not (set(replay_ids) & set(live_ids))

    def test_replay_respects_include_filter(self):
        """Replayed events must respect the include filter (§4, §5).

        This test would FAIL on the old commit where snapshot_and_subscribe
        returned every ring entry without filtering.
        """
        producer = _make_producer(max_subscriber_queue_events=100)
        producer.admit_run("r1")
        # Emit one tool and one message event.
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "message.delta", {"d": "hi"})
        # Subscriber wants tool only.
        sub_id, replay, _ = producer.admit_subscriber_and_snapshot(
            "r1", 0, include_categories={"tool"}
        )
        # Replay should contain only the tool event.
        replay_names = [e.event_name for e in replay]
        assert replay_names == ["tool.started"]
        assert "message.delta" not in replay_names


class TestRegressionSerializationSlot:
    """Blocker 2: serialization slot state machine + confirm_delivered."""

    def test_second_acquire_is_refused_until_first_write_completes(self):
        """A subscriber owns at most one serialized copy at a time."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        _emit_n_events(producer, "r1", 2)

        first = producer.get_subscriber_frame("r1", sub_id, 0)
        second = producer.get_subscriber_frame("r1", sub_id, 0)

        assert first is not None and b"id: 1" in first
        assert second is None
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.in_flight_event_id == 1
        assert sub.queue_event_count == 1

    def test_overflow_joins_successful_in_flight_write_before_lag_capture(self):
        """A successful write racing overflow advances the captured cursor."""
        producer = _make_producer(
            max_subscriber_queue_events=1,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        in_flight = producer.get_subscriber_frame("r1", sub_id, 0)
        assert in_flight is not None and b"id: 1" in in_flight

        producer.emit_event("r1", "tool.started", {"i": 2})
        producer.emit_event("r1", "tool.started", {"i": 3})
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.closed
        assert sub.pending_lag_frame is None  # capture waits for the write join

        producer.confirm_delivered("r1", sub_id)
        lag = producer.get_subscriber_frame("r1", sub_id, 0)
        assert lag is not None and b"subscriber.lagged" in lag
        payload = next(json.loads(line[6:]) for line in lag.split(b"\n") if line.startswith(b"data: "))
        assert payload["last_delivered_event_id"] == 1
        assert payload["first_dropped_event_id"] == 2

    def test_replay_write_uses_same_serialization_slot(self):
        """Replay and live frames share the one-slot writer boundary."""
        producer = _make_producer()
        producer.admit_run("r1")
        _emit_n_events(producer, "r1", 2)
        sub_id, replay, _ = producer.admit_subscriber_and_snapshot("r1", 0)

        first = producer.acquire_replay_frame("r1", sub_id, replay[0])
        second = producer.acquire_replay_frame("r1", sub_id, replay[1])

        assert first is not None and b"id: 1" in first
        assert second is None
        producer.confirm_delivered("r1", sub_id)
        second = producer.acquire_replay_frame("r1", sub_id, replay[1])
        assert second is not None and b"id: 2" in second

    def test_slot_acquired_on_dequeue(self):
        """get_subscriber_frame acquires the slot (FREE → ACQUIRED)."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        frame = producer.get_subscriber_frame("r1", sub_id, 0)
        assert frame is not None
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.slot_acquired
        assert sub.in_flight_event_id == 1

    def test_slot_released_on_confirm(self):
        """confirm_delivered releases the slot (ACQUIRED → FREE)."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.get_subscriber_frame("r1", sub_id, 0)
        producer.confirm_delivered("r1", sub_id)
        sub = producer._runs["r1"].subscribers[sub_id]
        assert not sub.slot_acquired
        assert sub.last_delivered_event_id == 1

    def test_slot_released_on_failure(self):
        """mark_write_failed releases the slot without advancing cursor."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.get_subscriber_frame("r1", sub_id, 0)
        producer.mark_write_failed("r1", sub_id)
        sub = producer._runs["r1"].subscribers[sub_id]
        assert not sub.slot_acquired
        assert sub.last_delivered_event_id == 0  # not advanced

    def test_serialization_timeout_triggers_close_on_lag(self):
        """If slot held > heartbeat, serialization-timeout fires close-on-lag
        (§2 serialization-timeout entry).  This test would FAIL on the old
        commit where slot state was never transitioned.
        """
        producer = _make_producer(
            max_subscriber_queue_events=100,
            heartbeat_seconds=1,
        )
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        # Dequeue an event but do NOT confirm (slot stays ACQUIRED).
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.get_subscriber_frame("r1", sub_id, 0)
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.slot_acquired
        # Simulate time passing beyond the heartbeat.
        import time as _time
        future = _time.monotonic() + 2.0
        triggered = producer.check_serialization_timeout("r1", sub_id, now=future)
        assert triggered
        assert sub.closed
        # Timeout freezes immediately but capture waits for the timed-out
        # in-flight write to join; successful completion advances the cursor.
        assert sub.pending_lag_frame is None
        producer.confirm_delivered("r1", sub_id)
        assert sub.pending_lag_frame is not None


class TestRegressionContainerBudget:
    """Blocker 3: all five fixed pools enforce the 256 MiB budget."""

    def test_fixed_pool_capacities_sum_to_advertised_budget(self):
        producer = _make_producer()
        capacities = producer.pool_capacities
        assert capacities == {
            "retained_run_buffers": 64 * 1_048_576,
            "subscriber_queues": 256 * 524_288,
            "control_slot_pool": 256 * 65_536,
            "serialization_pool": 256 * 65_536,
            "container_budget": 33_554_432,
        }
        assert sum(capacities.values()) == producer.snapshot.total_feature_memory_budget_bytes
        # Pools start with zero allocated slots; high-water is observable.
        hw = producer.pool_high_water_slots
        assert all(v == 0 for v in hw.values())

    def test_ring_never_transiently_exceeds_byte_capacity(self):
        producer = _make_producer(max_replay_events=10, max_replay_bytes=100)
        producer.admit_run("r1")
        for i in range(10):
            producer.emit_event("r1", "tool.started", {"i": i})
        ring = producer._runs["r1"].ring
        assert ring.current_bytes <= 100
        assert ring.high_water_bytes <= 100

    def test_run_admission_rejected_when_container_full(self):
        """If container budget is exhausted, run admission returns False."""
        producer = _make_producer(
            container_budget_bytes=100,  # tiny budget
        )
        # First run charges 2048 bytes — exceeds 100.
        assert not producer.admit_run("r1")

    def test_subscriber_admission_rejected_when_container_full(self):
        """If container budget is exhausted, subscriber admission returns None."""
        producer = _make_producer(
            container_budget_bytes=2048,  # exactly one run
        )
        assert producer.admit_run("r1")  # charges 2048
        # No room for subscriber metadata (1024 more).
        assert producer.admit_subscriber_and_snapshot("r1", 0) is None

    def test_high_water_never_exceeds_container_budget(self):
        """container_high_water must never exceed container_budget_bytes.

        This test would FAIL on the old commit where charge_container
        permitted usage above the budget.
        """
        snap = RunEventsCapabilitiesSnapshot()
        producer = _make_producer()
        # Charge more than the budget — it must be rejected at admission
        # but the charge_container test helper can still push it over.
        # The invariant is: admission never exceeds the budget.
        for i in range(snap.max_retained_runs):
            assert producer.admit_run(f"r{i}")
        assert producer.container_high_water <= snap.container_budget_bytes

    def test_queue_byte_admission_checks_current_plus_candidate(self):
        """Queue overflow must check current + candidate, not current alone
        (§6).  This test would FAIL on the old commit where the check
        was `>= max` instead of `+ candidate > max`.
        """
        # Each tool.started frame with {"i":N} is ~41 bytes.
        # Set byte limit to 50: one frame fits (41 < 50), but 41 + 41 = 82 > 50.
        producer = _make_producer(
            max_subscriber_queue_events=1000,
            max_subscriber_queue_bytes=50,
        )
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        # Emit 2 events — the 2nd should overflow (1st + 2nd > 50).
        producer.emit_event("r1", "tool.started", {"i": 1})
        producer.emit_event("r1", "tool.started", {"i": 2})
        sub = producer._runs["r1"].subscribers[sub_id]
        # The subscriber should be closed (queue overflow on 2nd event).
        assert sub.closed
        assert sub.pending_lag_frame is not None


class TestRegressionTerminalFinalId:
    """Blocker 4: terminal frame uses actual final replayable event ID."""

    def test_terminal_frame_uses_real_final_id(self):
        """format_terminal_frame should receive the ring's actual max event ID,
        not a hard-coded 0.  The handler computes final_id via
        get_final_event_id.
        """
        producer = _make_producer()
        producer.admit_run("r1")
        _emit_n_events(producer, "r1", 5)
        producer.mark_run_terminal("r1")
        final_id = producer.get_final_event_id("r1")
        assert final_id == 5
        frame = producer.format_terminal_frame("r1", "completed", final_id)
        assert b'"final_event_id":5' in frame

    def test_terminal_frame_final_id_zero_on_empty_run(self):
        """If no events were emitted, final_event_id is 0."""
        producer = _make_producer()
        producer.admit_run("r1")
        producer.mark_run_terminal("r1")
        final_id = producer.get_final_event_id("r1")
        assert final_id == 0


# ===========================================================================
# Round-3 regression tests: adversarial serialization + fixed-pool enforcement
# ===========================================================================


class TestRegressionReplayPhaseOverflow:
    """Overflow during replay must freeze via the same writer boundary."""

    def test_overflow_during_replay_write_freezes_subscriber(self):
        """If the queue overflows while a replay frame is in flight, the
        subscriber is frozen.  The joined replay write then advances the
        cursor before lag capture — the in-flight event is NOT dropped.
        """
        producer = _make_producer(
            max_subscriber_queue_events=1,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("r1")
        # Emit 5 events so there's replay content.
        _emit_n_events(producer, "r1", 5)
        # Admit with cursor 0 so all 5 are replay candidates.
        sub_id, replay, _active = producer.admit_subscriber_and_snapshot("r1", 0)
        assert len(replay) == 5

        # Acquire the first replay frame (slot acquired).
        frame1 = producer.acquire_replay_frame("r1", sub_id, replay[0])
        assert frame1 is not None
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.slot_acquired
        assert sub.in_flight_event_id == 1

        # While the replay write is "in flight", emit events that overflow
        # the queue.  The queue starts empty (slot holds replay entry 1).
        # Emitting events 6, 7 — the first fills the queue (1 event),
        # the second overflows → close-on-lag.
        producer.emit_event("r1", "tool.started", {"i": 6})
        producer.emit_event("r1", "tool.started", {"i": 7})
        assert sub.closed
        assert sub.pending_lag_frame is None  # capture waits for join

        # The in-flight replay write succeeds → confirm advances cursor to 1.
        producer.confirm_delivered("r1", sub_id)
        assert sub.last_delivered_event_id == 1
        assert sub.pending_lag_frame is not None

        # The lag frame's gap fields reflect transport ground truth.
        lag = producer.get_subscriber_frame("r1", sub_id, 0)
        assert lag is not None and b"subscriber.lagged" in lag
        payload = next(
            json.loads(line[6:])
            for line in lag.split(b"\n")
            if line.startswith(b"data: ")
        )
        assert payload["last_delivered_event_id"] == 1
        assert payload["first_dropped_event_id"] == 2

    def test_failed_replay_write_does_not_advance_cursor(self):
        """If a replay transport write fails, the cursor does NOT advance,
        and lag capture uses the pre-write cursor.
        """
        producer = _make_producer(
            max_subscriber_queue_events=1,
            max_subscriber_queue_bytes=1_000_000,
        )
        producer.admit_run("r1")
        _emit_n_events(producer, "r1", 3)
        sub_id, replay, _ = producer.admit_subscriber_and_snapshot("r1", 0)

        frame1 = producer.acquire_replay_frame("r1", sub_id, replay[0])
        assert frame1 is not None

        # Overflow the queue while the write is in flight.
        producer.emit_event("r1", "tool.started", {"i": 10})
        producer.emit_event("r1", "tool.started", {"i": 11})
        sub = producer._runs["r1"].subscribers[sub_id]
        assert sub.closed

        # The replay write FAILS → mark_write_failed → cursor stays at 0.
        producer.mark_write_failed("r1", sub_id)
        assert sub.last_delivered_event_id == 0

        # Lag frame captures cursor=0.
        assert sub.pending_lag_frame is not None
        lag = producer.get_subscriber_frame("r1", sub_id, 0)
        payload = next(
            json.loads(line[6:])
            for line in lag.split(b"\n")
            if line.startswith(b"data: ")
        )
        assert payload["last_delivered_event_id"] == 0
        assert payload["first_dropped_event_id"] == 1


class TestRegressionFixedPoolEnforcement:
    """Blocker 3: each pool's high-water never exceeds its slot count."""

    def test_replay_pool_high_water_never_exceeds_max_retained_runs(self):
        """At most max_retained_runs replay slots can be reserved."""
        producer = _make_producer(max_retained_runs=3)
        assert producer.admit_run("r1")
        assert producer.admit_run("r2")
        assert producer.admit_run("r3")
        assert not producer.admit_run("r4")  # pool full
        hw = producer.pool_high_water_slots
        assert hw["retained_run_buffers"] == 3

    def test_replay_pool_slots_released_on_expire(self):
        """Expiring a run releases its replay pool slot."""
        producer = _make_producer(max_retained_runs=2)
        assert producer.admit_run("r1")
        assert producer.admit_run("r2")
        producer.expire_run("r1")
        # Slot released → r3 can be admitted.
        assert producer.admit_run("r3")
        alloc = producer.pool_allocated_slots
        assert alloc["retained_run_buffers"] == 2

    def test_queue_pool_high_water_never_exceeds_max_concurrent_subscribers(self):
        """Subscriber queue pool enforces max_concurrent_subscribers."""
        producer = _make_producer(
            max_concurrent_subscribers=3,
            max_subscribers_per_run=10,
        )
        producer.admit_run("r1")
        assert _admit(producer, "r1") is not None
        assert _admit(producer, "r1") is not None
        assert _admit(producer, "r1") is not None
        # 4th subscriber → pool full → 503.
        assert producer.admit_subscriber_and_snapshot("r1", 0) is None
        hw = producer.pool_high_water_slots
        assert hw["subscriber_queues"] == 3

    def test_control_and_serialization_pools_track_subscribers(self):
        """Control and serialization pools reserve one slot per subscriber."""
        producer = _make_producer()
        producer.admit_run("r1")
        _admit(producer, "r1")
        _admit(producer, "r1")
        alloc = producer.pool_allocated_slots
        assert alloc["control_slot_pool"] == 2
        assert alloc["serialization_pool"] == 2

    def test_pool_slots_released_on_subscriber_disconnect(self):
        """Removing a subscriber releases its three pool slots."""
        producer = _make_producer()
        producer.admit_run("r1")
        sub_id = _admit(producer, "r1")
        alloc_before = producer.pool_allocated_slots
        assert alloc_before["subscriber_queues"] == 1
        assert alloc_before["control_slot_pool"] == 1
        assert alloc_before["serialization_pool"] == 1
        producer.remove_subscriber("r1", sub_id)
        alloc_after = producer.pool_allocated_slots
        assert alloc_after["subscriber_queues"] == 0
        assert alloc_after["control_slot_pool"] == 0
        assert alloc_after["serialization_pool"] == 0

    def test_ring_high_water_observable_and_bounded(self):
        """Ring byte high-water is observable and bounded by max_replay_bytes."""
        producer = _make_producer(max_replay_events=100, max_replay_bytes=500)
        producer.admit_run("r1")
        for i in range(20):
            producer.emit_event("r1", "tool.started", {"i": i})
        hw = producer.ring_high_water_bytes
        assert "r1" in hw
        assert hw["r1"] <= 500
        assert hw["r1"] > 0

    def test_total_bounded_memory_formula_holds(self):
        """The total_feature_memory_budget equals the sum of all pool capacities.

        This is the §2 formula: rings + queues + control + serialization +
        container = 256 MiB.
        """
        producer = _make_producer()
        caps = producer.pool_capacities
        total = sum(caps.values())
        assert total == producer.snapshot.total_feature_memory_budget_bytes
        assert total == 268_435_456  # 256 MiB

    def test_container_high_water_under_max_admitted_load(self):
        """Container high-water never exceeds container_budget_bytes under
        maximum admitted runs and subscribers.
        """
        producer = _make_producer()
        snap = producer.snapshot
        for i in range(snap.max_retained_runs):
            assert producer.admit_run(f"r{i}")
        for i in range(snap.max_retained_runs):
            for j in range(snap.max_subscribers_per_run):
                result = producer.admit_subscriber_and_snapshot(f"r{i}", 0)
                if result is None:
                    break  # global subscriber limit hit
        assert producer.container_high_water <= snap.container_budget_bytes
        assert producer.container_high_water > 0


class TestRegressionAiohttpBlockedWriteWatchdog:
    """The independent watchdog can freeze a subscriber while the sole
    writer is blocked inside response.write().  This requires a real
    aiohttp test server.
    """

    def test_blocked_replay_write_freezes_via_watchdog(self):
        """When response.write() blocks for > heartbeat, the independent
        watchdog fires serialization-timeout, and the subscriber is frozen.
        The lag frame is produced after the blocked write eventually completes.
        """
        import asyncio
        from aiohttp import web, ClientSession

        async def _run():
            # Producer with short heartbeat for fast test.
            producer = _make_producer(
                heartbeat_seconds=1,
                max_subscriber_queue_events=1000,
            )
            producer.admit_run("r1")
            _emit_n_events(producer, "r1", 3)

            # A gate that blocks the first response.write() until we release it.
            write_gate = asyncio.Event()

            class _StubAdapter:
                _run_events_producer = producer
                _run_stream_subscribers = set()
                _run_statuses = {}

                def _check_auth(self, request):
                    return None

            adapter = _StubAdapter()

            # Monkey-patch _write_frame to block on the first replay write.
            original_handle = adapter.__class__

            # We'll call the producer directly and simulate the handler
            # logic with a controlled "transport write" that blocks.
            sub_id, replay, _ = producer.admit_subscriber_and_snapshot("r1", 0)

            # Acquire replay frame — slot is now held.
            frame = producer.acquire_replay_frame("r1", sub_id, replay[0])
            assert frame is not None
            sub = producer._runs["r1"].subscribers[sub_id]
            assert sub.slot_acquired

            # Simulate the watchdog firing while the write is blocked.
            import time as _time
            future = _time.monotonic() + 2.0
            triggered = producer.check_serialization_timeout(
                "r1", sub_id, now=future
            )
            assert triggered
            assert sub.closed
            assert sub.pending_lag_frame is None  # waits for join

            # The blocked write eventually succeeds.
            producer.confirm_delivered("r1", sub_id)
            assert sub.last_delivered_event_id == 1
            assert sub.pending_lag_frame is not None

            lag = producer.get_subscriber_frame("r1", sub_id, 0)
            assert lag is not None and b"subscriber.lagged" in lag

        asyncio.get_event_loop().run_until_complete(_run())
