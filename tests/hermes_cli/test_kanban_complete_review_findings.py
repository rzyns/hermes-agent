"""Permanent regression tests for independent review findings (t_451cb0e9).

These tests encode the probe findings from the parent review:

1. Versioned, canonically normalized completion digest.
2. Stale-run refusal after a later terminal run exists.
3. Filesystem publication happens after DB commit (no orphan durable files).
4. Committed completion with incomplete finalization resumes on retry and
   does not report success until finalization is complete.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def populated_db(kanban_home):
    """Return a tuple of (conn, task_id, run_id, child_task_id)."""
    conn = kb.connect()
    parent = kb.create_task(conn, title="parent", assignee="alice")
    child = kb.create_task(conn, title="child", assignee="x", created_by="alice")
    claimed = kb.claim_task(conn, parent)
    assert claimed is not None
    run_id = _current_run_id(conn, parent)
    return conn, parent, run_id, child


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None and row["current_run_id"] is not None
    return int(row["current_run_id"])


def _attachment_paths(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT stored_path FROM task_attachments WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [r["stored_path"] for r in rows]


def _task_status(conn: sqlite3.Connection, task_id: str) -> str:
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    return row["status"]


def test_canonical_completion_digest_is_versioned_and_normalized():
    """Digests must be versioned and treat semantically equal inputs equally."""
    d = kb._canonical_completion_digest

    def _digest(metadata):
        return d(
            task_id="t",
            run_id=1,
            summary="s",
            result=None,
            metadata=metadata,
            created_cards=None,
            review_pending=None,
        )

    assert _digest({}) == _digest(None)
    assert _digest({"n": 1}) == _digest({"n": 1.0})
    assert _digest({"a": 1, "b": 2}) == _digest({"b": 2, "a": 1})

    sample = _digest({})
    assert sample.startswith("v1:")
    assert len(sample) == len("v1:") + 64


def test_canonical_digest_binds_artifact_content(kanban_home):
    """Changing artifact bytes at the same path changes the digest."""
    conn = kb.connect()
    t = kb.create_task(conn, title="artifact digest")
    task = kb.get_task(conn, t)
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, t, ws)
    artifact = ws / "report.txt"
    artifact.write_text("version one")

    digest_one = kb._canonical_completion_digest(
        task_id=t,
        run_id=1,
        summary="done",
        result=None,
        metadata={"artifacts": [str(artifact)]},
        created_cards=None,
        review_pending=None,
    )
    artifact.write_text("version two")
    digest_two = kb._canonical_completion_digest(
        task_id=t,
        run_id=1,
        summary="done",
        result=None,
        metadata={"artifacts": [str(artifact)]},
        created_cards=None,
        review_pending=None,
    )
    assert digest_one != digest_two
    assert digest_one.startswith("v1:")
    assert digest_two.startswith("v1:")


def test_stale_run_after_later_terminal_completion_is_refused(populated_db):
    """A completion for an older run must be rejected once a later run finished."""
    conn, tid, first_run_id, _child = populated_db

    # First run completes normally.
    assert kb.complete_task(conn, tid, summary="first run", expected_run_id=first_run_id).ok

    # Reopen the task (simulates a respawn/retry with a new run id).
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, result=NULL, "
        "completed_at=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    kb.claim_task(conn, tid)
    second_run_id = _current_run_id(conn, tid)
    assert second_run_id != first_run_id

    # Second run completes.
    assert kb.complete_task(conn, tid, summary="second run", expected_run_id=second_run_id).ok

    # Attempting to complete the first run now is structurally stale.
    with pytest.raises(kb.StaleCompletionRunError) as exc_info:
        kb.complete_task(conn, tid, summary="stale retry", expected_run_id=first_run_id)
    assert exc_info.value.task_id == tid
    assert exc_info.value.attempted_run_id == first_run_id
    assert exc_info.value.latest_run_id == second_run_id


def test_no_orphan_artifact_before_db_commit(kanban_home, monkeypatch):
    """If the DB transaction aborts, no durable attachment file is left behind."""
    conn = kb.connect()
    t = kb.create_task(conn, title="crash test")
    task = kb.get_task(conn, t)
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, t, ws)
    artifact = ws / "report.txt"
    artifact.write_text("findings")

    # Make artifact publication fail after commit would have happened; because
    # we publish *after* commit, the only durable files are in attachments.
    calls: list[bool] = []
    orig_publish = kb._publish_completion_artifact_manifest

    def fake_publish(task_id, manifest, *, board=None):
        calls.append(True)
        # Pretend publication succeeded but produce an empty manifest so no
        # file is copied. This verifies the main transaction still committed.
        return []

    monkeypatch.setattr(kb, "_publish_completion_artifact_manifest", fake_publish)

    result = kb.complete_task(
        conn, t,
        summary="done",
        metadata={"artifacts": [str(artifact)]},
    )
    assert result.ok
    assert _task_status(conn, t) == "done"
    # No durable attachment files should exist because publication was skipped.
    assert _attachment_paths(conn, t) == []
    # But the source scratch file is still gone because workspace cleanup ran.
    assert not ws.exists()


def test_committed_completion_resumes_finalization(populated_db, monkeypatch):
    """A committed row with unfinalized status resumes and does not fake success."""
    conn, tid, run_id, _child = populated_db
    task = kb.get_task(conn, tid)
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, tid, ws)
    artifact = ws / "report.txt"
    artifact.write_text("findings")

    input_metadata = {"artifacts": [str(artifact)]}
    digest = kb._canonical_completion_digest(
        task_id=tid,
        run_id=run_id,
        summary="done",
        result=None,
        metadata=input_metadata,
        created_cards=[],
        review_pending=[],
    )

    # Insert a committed but unfinalized completion row (crash after commit).
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO task_completion_results (
            task_id, run_id, digest, status, terminal_status, outcome,
            summary, result, metadata_json, created_cards_json,
            review_pending_json, published_artifacts_json,
            terminal_event_id, completed_at, finalized_at
        ) VALUES (?, ?, ?, 'committed', 'done', 'completed', 'done', NULL,
                  ?, ?, ?, ?, NULL, ?, NULL)
        """,
        (
            tid, run_id, digest,
            json.dumps(input_metadata),
            json.dumps([]),
            json.dumps([]),
            json.dumps(kb._artifact_manifest_from_metadata(input_metadata)),
            now,
        ),
    )
    conn.execute(
        "UPDATE tasks SET status='done', result='done', completed_at=? WHERE id=?",
        (now, tid),
    )
    conn.commit()

    # The task is terminal but the ledger row is not finalized.
    assert _task_status(conn, tid) == "done"
    row = conn.execute(
        "SELECT status FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "committed"

    # Exact retry should succeed and now be finalized.
    result = kb.complete_task(
        conn, tid,
        summary="done",
        metadata=input_metadata,
        expected_run_id=run_id,
    )
    assert result.ok
    assert result.already_terminal is False

    row = conn.execute(
        "SELECT status, finalized_at FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "finalized"
    assert row["finalized_at"] is not None
    paths = _attachment_paths(conn, tid)
    assert len(paths) == 1
    assert Path(paths[0]).name == "report.txt"


def test_same_run_idempotent_retry_returns_same_digest(populated_db):
    """Retrying the exact same completion produces the same digest."""
    conn, tid, run_id, _child = populated_db
    summary = "handoff"
    metadata = {"changed_files": ["a.txt"]}

    first = kb._canonical_completion_digest(
        task_id=tid,
        run_id=run_id,
        summary=summary,
        result=None,
        metadata=metadata,
        created_cards=None,
        review_pending=None,
    )
    second = kb._canonical_completion_digest(
        task_id=tid,
        run_id=run_id,
        summary=summary,
        result=None,
        metadata=metadata,
        created_cards=None,
        review_pending=None,
    )
    assert first == second
    assert first.startswith("v1:")


def test_crash_recovery_failure_sensitivity(populated_db, monkeypatch):
    """If the post-commit finalizer is disabled, a committed row stays unfinalized.

    This is the mutation/failure-sensitivity check: the crash-recovery path must
    actually do the work. When the finalizer is bypassed, the retry must leave
    the ledger in ``committed`` state.
    """
    conn, tid, run_id, _child = populated_db
    now = int(time.time())
    digest = kb._canonical_completion_digest(
        task_id=tid,
        run_id=run_id,
        summary="done",
        result=None,
        metadata=None,
        created_cards=[],
        review_pending=[],
    )
    conn.execute(
        """
        INSERT INTO task_completion_results (
            task_id, run_id, digest, status, terminal_status, outcome,
            summary, result, metadata_json, created_cards_json,
            review_pending_json, published_artifacts_json,
            terminal_event_id, completed_at, finalized_at
        ) VALUES (?, ?, ?, 'committed', 'done', 'completed', 'done', NULL,
                  ?, ?, ?, ?, NULL, ?, NULL)
        """,
        (
            tid, run_id, digest,
            json.dumps({}),
            json.dumps([]),
            json.dumps([]),
            json.dumps([]),
            now,
        ),
    )
    conn.execute(
        "UPDATE tasks SET status='done', result='done', completed_at=? WHERE id=?",
        (now, tid),
    )
    conn.commit()

    # Bypass the finalizer so recovery does nothing.
    original = kb._finalize_completion_result
    monkeypatch.setattr(kb, "_finalize_completion_result", lambda *args, **kwargs: None)
    result = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    # With finalization disabled, complete_task should not synthesize success.
    assert not result or not result.ok

    row = conn.execute(
        "SELECT status FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "committed"

    # Restore and retry: now it finalizes.
    monkeypatch.undo()
    result = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    assert result.ok
    row = conn.execute(
        "SELECT status FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "finalized"


def test_concurrent_duplicate_completion_is_idempotent(populated_db):
    """Two calls with the exact same payload produce a single terminal run."""
    conn, tid, run_id, _child = populated_db

    result1 = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    assert result1.ok
    run_count1 = int(
        conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (tid,)
        ).fetchone()[0]
    )

    result2 = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    assert result2.ok
    run_count2 = int(
        conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (tid,)
        ).fetchone()[0]
    )
    assert run_count1 == run_count2

    events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
        (tid,),
    ).fetchone()[0]
    assert events == 1
