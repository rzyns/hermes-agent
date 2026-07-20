"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------


def test_first_typed_block_lands_in_blocked(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="which key?", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind == "needs_input"
        assert t.block_recurrences == 1


def test_unblock_does_not_reset_recurrence_counter(kanban_home: Path) -> None:
    """The crux of the fix: unblock must preserve the loop counter."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="needs_input")
        assert kb.get_task(conn, tid).block_recurrences == 1
        assert kb.unblock_task(conn, tid)
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.block_recurrences == 1  # NOT reset to 0
        assert t.block_kind == "needs_input"  # kind preserved for comparison


def test_same_cause_reblock_routes_to_triage(kanban_home: Path) -> None:
    """Dale's loop: block → unblock → re-block same kind → triage."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="still need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        assert t.block_recurrences == 2


def test_untyped_block_loop_also_protected(kanban_home: Path) -> None:
    """Legacy un-typed blocks (kind=None) still trip the breaker."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="a again")
        assert kb.get_task(conn, tid).status == "triage"


def test_different_kinds_do_not_compound(kanban_home: Path) -> None:
    """A re-block for a DIFFERENT reason resets the counter to 1."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="b", kind="capability")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_recurrences == 1


def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_block_routes_to_todo(kanban_home: Path) -> None:
    """Dependency waits never enter the human 'blocked' bucket."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="need X first", kind="dependency")
        t = kb.get_task(conn, tid)
        assert t.status == "todo"
        assert t.block_kind == "dependency"


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


def test_completion_clears_block_memory(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(
            conn,
            tid,
            reason="x",
            kind="capability",
            contract={
                "error_type": "provider_auth",
                "fingerprint": "auth:provider-a",
                "retry_after": 60,
            },
        )
        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).block_recurrences == 1
        kb.complete_task(conn, tid, result="done")
        t = kb.get_task(conn, tid)
        assert t.status == "done"
        assert t.block_recurrences == 0
        assert t.block_kind is None
        row = conn.execute(
            "SELECT block_fingerprint, block_deadline, block_retry_after, "
            "block_error_type FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert tuple(row) == (None, None, None, None)


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


def test_invalid_kind_rejected(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, reason="x", kind="bogus")


def test_block_without_kind_is_backward_compatible(kanban_home: Path) -> None:
    """Existing callers that pass no kind keep the old single-block behaviour."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="legacy")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind is None


# ---------------------------------------------------------------------------
# Structured failure contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error_type", "expected_kind", "expected_retry_after", "deadline_delta"),
    [
        ("provider_quota", "infra", None, 4 * 60 * 60),
        ("provider_auth", "infra", None, 4 * 60 * 60),
        ("model_missing", "infra", None, 4 * 60 * 60),
        ("rate_limit", "transient", 30 * 60, 30 * 60),
        ("budget_exhausted", "budget", None, 24 * 60 * 60),
    ],
)
def test_failure_contract_maps_error_type_and_stamps_class_deadline(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: str,
    expected_kind: str,
    expected_retry_after: int | None,
    deadline_delta: int,
) -> None:
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(
            conn,
            tid,
            contract={"error_type": error_type, "detail": "structured failure"},
        )

        row = conn.execute(
            "SELECT block_kind, block_error_type, block_retry_after, "
            "block_deadline FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["block_kind"] == expected_kind
        assert row["block_error_type"] == error_type
        assert row["block_retry_after"] == expected_retry_after
        assert row["block_deadline"] == now + deadline_delta

        event = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert event.payload["contract"]["error_type"] == error_type
        assert event.payload["contract"]["detail"] == "structured failure"


def test_rate_limit_contract_honors_explicit_retry_after(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(
            conn,
            tid,
            contract={
                "error_type": "rate_limit",
                "retryable": True,
                "retry_after": 90,
                "detail": "provider asked us to wait",
            },
        )
        row = conn.execute(
            "SELECT block_retry_after, block_deadline FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["block_retry_after"] == 90
        assert row["block_deadline"] == now + 90
        event = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert event.payload["contract"] == {
            "error_type": "rate_limit",
            "fingerprint": None,
            "retryable": True,
            "retry_after": 90,
            "needs_authority": False,
            "detail": "provider asked us to wait",
        }


def test_explicit_kind_wins_over_failure_contract_mapping(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(
            conn,
            tid,
            reason="human must choose",
            kind="needs_input",
            contract={"error_type": "provider_quota"},
        )
        row = conn.execute(
            "SELECT block_kind, block_deadline FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["block_kind"] == "needs_input"
        assert row["block_deadline"] is None


def test_bare_reason_is_recorded_as_judgment_contract(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="legacy judgment")
        row = conn.execute(
            "SELECT block_error_type, block_deadline FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["block_error_type"] == "judgment"
        assert row["block_deadline"] is None
        event = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1]
        assert event.payload["contract"] == {
            "error_type": "judgment",
            "fingerprint": None,
            "retryable": False,
            "retry_after": None,
            "needs_authority": False,
            "detail": "legacy judgment",
        }
