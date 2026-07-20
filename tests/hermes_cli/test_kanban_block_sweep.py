"""Behavioral coverage for frozen-subtree and block-deadline reconciliation."""

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


def _running_task(conn, title: str) -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    return task_id


def _events(conn, task_id: str, kind: str):
    return [event for event in kb.list_events(conn, task_id) if event.kind == kind]


def test_block_sweep_freezes_distinct_todo_and_scheduled_descendants(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb.time, "time", lambda: 1_800_000_000)
    with kb.connect_closing() as conn:
        root = _running_task(conn, "infra root")
        left = kb.create_task(conn, title="left")
        right = kb.create_task(conn, title="right")
        leaf = kb.create_task(conn, title="leaf")
        kb.link_tasks(conn, root, left)
        kb.link_tasks(conn, root, right)
        kb.link_tasks(conn, left, leaf)
        kb.link_tasks(conn, right, leaf)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'scheduled' WHERE id = ?", (right,))
        assert kb.block_task(conn, root, reason="provider unavailable", kind="infra")

        frozen_roots, deadlines = kb.sweep_block_propagation_and_deadlines(
            conn, now=1_800_000_000
        )

        assert frozen_roots == [root]
        assert deadlines == []
        root_event = _events(conn, root, "subtree_frozen")[-1]
        assert root_event.payload == {
            "cause_task": root,
            "class": "infra",
            "frozen_count": 3,
        }
        for descendant in (left, right, leaf):
            annotation = _events(conn, descendant, "frozen_by")[-1]
            assert annotation.payload == {"cause_task": root, "class": "infra"}


def test_block_sweep_reemits_only_delta_when_late_child_is_added(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb.time, "time", lambda: 1_800_000_000)
    with kb.connect_closing() as conn:
        root = _running_task(conn, "budget root")
        first = kb.create_task(conn, title="first")
        kb.link_tasks(conn, root, first)
        assert kb.block_task(conn, root, reason="turn budget exhausted", kind="budget")
        kb.sweep_block_propagation_and_deadlines(conn, now=1_800_000_000)

        late = kb.create_task(conn, title="late")
        kb.link_tasks(conn, first, late)
        frozen_roots, deadlines = kb.sweep_block_propagation_and_deadlines(
            conn, now=1_800_000_001
        )

        assert frozen_roots == [root]
        assert deadlines == []
        assert [event.payload["frozen_count"] for event in _events(conn, root, "subtree_frozen")] == [1, 2]
        assert len(_events(conn, first, "frozen_by")) == 1
        assert len(_events(conn, late, "frozen_by")) == 1


def test_block_sweep_emits_class_deadline_dispositions_without_cancelling(
    kanban_home: Path,
) -> None:
    now = 1_800_000_000
    with kb.connect_closing() as conn:
        infra = _running_task(conn, "infra")
        budget = _running_task(conn, "budget")
        triage = kb.create_task(conn, title="triage", triage=True)
        assert kb.block_task(conn, infra, reason="provider unavailable", kind="infra")
        assert kb.block_task(conn, budget, reason="budget exhausted", kind="budget")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_deadline = ? WHERE id IN (?, ?, ?)",
                (now, infra, budget, triage),
            )

        _, expired = kb.sweep_block_propagation_and_deadlines(conn, now=now)

        assert expired == sorted([budget, infra, triage])
        expected = {
            infra: ("infra", "escalate"),
            budget: ("budget", "re-notify"),
            triage: ("triage", "re-notify"),
        }
        for task_id, (block_class, disposition) in expected.items():
            event = _events(conn, task_id, "deadline_expired")[-1]
            assert event.payload == {
                "class": block_class,
                "disposition": disposition,
                "deadline": now,
            }
        assert kb.get_task(conn, infra).status == "blocked"
        assert kb.get_task(conn, budget).status == "blocked"
        assert kb.get_task(conn, triage).status == "triage"


def test_block_sweep_emits_no_events_when_reconciled_state_is_unchanged(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect_closing() as conn:
        root = _running_task(conn, "infra")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, root, child)
        assert kb.block_task(conn, root, reason="provider unavailable", kind="infra")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET block_deadline = ? WHERE id = ?", (now, root))
        first = kb.dispatch_once(conn, max_spawn=0)
        assert first.frozen_subtrees == [root]
        assert first.expired_deadlines == [root]
        event_count = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]

        second = kb.dispatch_once(conn, max_spawn=0)

        assert second.frozen_subtrees == []
        assert second.expired_deadlines == []
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == event_count
