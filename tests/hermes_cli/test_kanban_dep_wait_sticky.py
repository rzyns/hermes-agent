"""Regression tests for the sticky dependency-wait gate (DISP-1).

The 2026-07-27 incident: two cards hot-looped against an unchanged
dependency for hours — ``dependency_wait`` → ``promoted`` → ``claimed`` →
``spawned`` → ``dependency_wait`` — burning ~122 worker spawns for zero
forward progress. The circuit breaker (``consecutive_failures``) never
tripped because a dependency_wait is not a failure; ``block_recurrences``
never accumulated because dependency blocks route to ``todo`` (not
``blocked``), never passing through ``unblock_task``.

The fix: ``block_task(kind="dependency")`` now stamps a
``dep_wait_fingerprint`` (a hash of parent ids + status + completed_at).
``recompute_ready`` compares that stored fingerprint against a fresh
computation and refuses to re-promote when they match. The task stays
parked in ``todo`` until the dependency materially changes (parent
completes, link removed, etc.).

Backstop: ``DEP_WAIT_REDISPATCH_CAP`` limits consecutive dependency_wait
re-dispatches per task before emitting a deduplicated
``dependency_wait_backstop`` event.
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


def _force_running_for_reblock(conn, tid):
    """Force a task from todo (dependency-parked) back to running WITHOUT
    clearing dep_wait fields.

    This simulates the exact buggy-dispatcher scenario: the task was
    dependency-waiting in todo, then (incorrectly) promoted to ready,
    claimed, and is now running again — all without the dep_wait
    fingerprint being cleared (which is what the fixed recompute_ready
    now does, but we want to test the block_task counting independently).

    We set status='running' directly so ``block_task``'s
    ``status IN ('running', 'ready')`` guard passes.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='running', "
            "claim_lock='test', claim_expires=NULL "
            "WHERE id=?",
            (tid,),
        )


# ---------------------------------------------------------------------------
# Sticky dependency-wait gate
# ---------------------------------------------------------------------------


def test_dependency_wait_sets_fingerprint(kanban_home: Path) -> None:
    """block_task(kind=dependency) stamps a dep_wait_fingerprint."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="waiting", kind="dependency")
        t = kb.get_task(conn, child)
        assert t.status == "todo"
        assert t.dep_wait_fingerprint is not None
        assert t.dep_wait_count == 1


def test_unchanged_dependency_not_repromoted(kanban_home: Path) -> None:
    """Core fix: a dependency_wait against an unchanged parent is NOT
    re-promoted on subsequent recompute_ready ticks."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Worker blocks — parent is not done.
        assert kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Simulate the dispatcher ticking recompute_ready N times.
        for _ in range(5):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "todo", (
                "child was re-promoted while dependency was unchanged — "
                "the sticky gate failed"
            )


def test_promoted_once_parent_genuinely_completes(kanban_home: Path) -> None:
    """The gate releases once the parent genuinely completes — the child
    IS promoted on the next recompute_ready tick."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Parent completes — fingerprint changes.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready' WHERE id=?", (parent,)
            )
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        # dep_wait fields should be cleared on promotion.
        child_t = kb.get_task(conn, child)
        assert child_t.dep_wait_fingerprint is None
        assert child_t.dep_wait_count == 0


def test_repeated_dependency_blocks_increment_count(kanban_home: Path) -> None:
    """Each dependency_wait block for the SAME unchanged condition
    increments dep_wait_count (the signal the backstop keys on)."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # First dependency_wait.
        kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.get_task(conn, child).dep_wait_count == 1
        # Simulate a (buggy) re-promotion and re-block.
        _force_running_for_reblock(conn, child)
        kb.block_task(conn, child, reason="still waiting", kind="dependency")
        t = kb.get_task(conn, child)
        assert t.dep_wait_count == 2, (
            f"expected dep_wait_count=2 for same-fingerprint re-block, "
            f"got {t.dep_wait_count}"
        )
        assert t.status == "todo"


# ---------------------------------------------------------------------------
# Backstop event
# ---------------------------------------------------------------------------


def test_backstop_event_fires_and_deduplicated(kanban_home: Path) -> None:
    """When dep_wait_count reaches the cap, a ``dependency_wait_backstop``
    event is emitted. Calling the backstop check again on the same
    fingerprint/count does NOT emit a duplicate."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Drive dep_wait_count to the cap by repeatedly blocking.
        for _ in range(kb.DEP_WAIT_REDISPATCH_CAP):
            _force_running_for_reblock(conn, child)
            kb.block_task(conn, child, reason="waiting", kind="dependency")
        t = kb.get_task(conn, child)
        assert t.dep_wait_count >= kb.DEP_WAIT_REDISPATCH_CAP
        # Crossing the cap emits immediately in block_task, independent of
        # whether a later dispatcher tick happens.
        events = [
            e for e in kb.list_events(conn, child)
            if e.kind == "dependency_wait_backstop"
        ]
        assert len(events) == 1, (
            f"expected exactly 1 backstop event, got {len(events)}"
        )
        payload = events[0].payload or {}
        assert payload.get("cap") == kb.DEP_WAIT_REDISPATCH_CAP
        # Tick again — no duplicate event for the same count.
        kb.recompute_ready(conn)
        events = [
            e for e in kb.list_events(conn, child)
            if e.kind == "dependency_wait_backstop"
        ]
        assert len(events) == 1, (
            "backstop event was duplicated on a second tick for the "
            "same fingerprint/count — dedup failed"
        )


def test_dispatch_result_surfaces_stalled_dependency_wait(
    kanban_home: Path,
) -> None:
    """dispatch_once reports tasks held by the sticky dependency-wait gate
    in ``DispatchResult.dependency_wait_stalled`` for telemetry."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="waiting", kind="dependency")
        result = kb.dispatch_once(conn, spawn_fn=lambda *_: None)
        assert child in result.dependency_wait_stalled
        assert kb.get_task(conn, child).status == "todo"


def test_archived_not_completed_parent_stays_gated(kanban_home: Path) -> None:
    """The exact incident scenario: parent was archived with
    completed_at=NULL. The child must not be promoted even though
    _parent_dependency_satisfied correctly returns False for this parent
    — the dispatcher and worker may disagree about whether such a parent
    satisfies the dependency, so the sticky gate must hold."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Simulate the incident: parent archived without completing.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', completed_at=NULL "
                "WHERE id=?",
                (parent,),
            )
        # Worker blocks on dependency.
        assert kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # recompute_ready should NOT promote (fingerprint is unchanged).
        for _ in range(3):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "todo"


def test_complete_clears_dep_wait_fields(kanban_home: Path) -> None:
    """Completing a task clears its own dep_wait fields."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="waiting", kind="dependency")
        assert kb.get_task(conn, tid).dep_wait_fingerprint is not None
        # Move back to running for completion.
        _force_running_for_reblock(conn, tid)
        kb.complete_task(conn, tid, result="done")
        t = kb.get_task(conn, tid)
        assert t.status == "done"
        assert t.dep_wait_fingerprint is None
        assert t.dep_wait_count == 0


def test_unblock_clears_dep_wait_fields(kanban_home: Path) -> None:
    """Unblocking a task that had a dependency_wait clears the dep_wait
    fingerprint so the task gets a fresh start."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        # Dependency-block routes to todo; explicit operator unblock must
        # nevertheless release the sticky gate directly.
        kb.block_task(conn, tid, reason="waiting", kind="dependency")
        assert kb.get_task(conn, tid).dep_wait_fingerprint is not None
        assert kb.unblock_task(conn, tid)
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.dep_wait_fingerprint is None
        assert t.dep_wait_count == 0


def test_unblock_sticky_wait_keeps_open_parent_gate(kanban_home: Path) -> None:
    """Operator override clears stickiness without bypassing task links."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="waiting", kind="dependency")

        assert kb.unblock_task(conn, child)
        task = kb.get_task(conn, child)
        assert task.status == "todo"
        assert task.dep_wait_fingerprint is None
        assert task.dep_wait_count == 0
