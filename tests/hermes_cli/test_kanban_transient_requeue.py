"""Behavioral coverage for config-gated transient block requeues."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.config import DEFAULT_CONFIG


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title: str = "transient") -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    return task_id


def _block_transient(conn, task_id: str) -> None:
    assert kb.block_task(
        conn,
        task_id,
        contract={
            "error_type": "rate_limit",
            "retryable": True,
            "detail": "temporary provider throttle",
        },
    )


def test_retry_config_defaults_off() -> None:
    assert DEFAULT_CONFIG["kanban"]["retry"]["enabled"] is False


def test_transient_requeue_waits_until_deadline_then_requeues(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        _block_transient(conn, task_id)

        requeued, escalated = kb.requeue_due_transients(conn, now=now + 1799)
        assert requeued == []
        assert escalated == []
        assert kb.get_task(conn, task_id).status == "blocked"

        requeued, escalated = kb.requeue_due_transients(conn, now=now + 1800)
        assert requeued == [task_id]
        assert escalated == []
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.block_recurrences == 1


def test_transient_retry_uses_exponential_backoff_capped_at_four_hours(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_800_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        _block_transient(conn, task_id)
        first = conn.execute(
            "SELECT block_retry_after, block_deadline FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(first) == (30 * 60, clock[0] + 30 * 60)
        first_task = kb.get_task(conn, task_id)
        assert first_task is not None
        assert first_task.block_recurrences == 0

        kb.requeue_due_transients(conn, now=clock[0] + 30 * 60)
        clock[0] += 30 * 60
        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        _block_transient(conn, task_id)
        second = conn.execute(
            "SELECT block_retry_after, block_deadline FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(second) == (60 * 60, clock[0] + 60 * 60)

        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', block_recurrences = 10 WHERE id = ?",
                (task_id,),
            )
        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        _block_transient(conn, task_id)
        capped = conn.execute(
            "SELECT block_retry_after FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        assert capped["block_retry_after"] == 4 * 60 * 60


def test_second_due_transient_escalates_to_triage(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_800_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    with kb.connect_closing() as conn:
        task_id = _running_task(conn)
        _block_transient(conn, task_id)
        kb.requeue_due_transients(conn, now=clock[0] + 30 * 60)

        clock[0] += 30 * 60
        assert kb.claim_task(conn, task_id, claimer="worker") is not None
        _block_transient(conn, task_id)
        requeued, escalated = kb.requeue_due_transients(conn, now=clock[0] + 60 * 60)

        assert requeued == []
        assert escalated == [task_id]
        task = kb.get_task(conn, task_id)
        assert task.status == "triage"
        assert task.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT


def test_dispatch_retry_flag_off_is_noop_and_never_touches_other_block_kinds(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [1_800_000_000]
    monkeypatch.setattr(kb.time, "time", lambda: clock[0])
    with kb.connect_closing() as conn:
        transient = _running_task(conn, "transient")
        _block_transient(conn, transient)
        needs_input = _running_task(conn, "needs-input")
        assert kb.block_task(conn, needs_input, reason="human choice", kind="needs_input")
        legacy = _running_task(conn, "legacy")
        assert kb.block_task(conn, legacy, reason="legacy unknown")
        # Give the protected rows elapsed deadlines too: kind, not deadline
        # absence, is the fail-closed boundary.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_deadline = ? WHERE id IN (?, ?)",
                (clock[0], needs_input, legacy),
            )

        kb.dispatch_once(conn, retry_enabled=False, max_spawn=0)
        assert kb.get_task(conn, transient).status == "blocked"

        clock[0] += 30 * 60
        result = kb.dispatch_once(conn, retry_enabled=True, max_spawn=0)
        assert result.requeued_transients == [transient]
        assert kb.get_task(conn, transient).status == "ready"
        assert kb.get_task(conn, needs_input).status == "blocked"
        assert kb.get_task(conn, legacy).status == "blocked"
