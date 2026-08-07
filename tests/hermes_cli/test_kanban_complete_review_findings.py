"""Permanent regression tests for independent review findings (t_451cb0e9 / t_551bb31a).

These tests encode the probe findings from the parent reviews:

1. Versioned, canonically normalized completion digest that is content-bound,
   not path-bound.
2. Stale-run refusal after a later terminal run exists.
3. Filesystem publication happens after DB commit (no orphan durable files).
4. Committed completion with incomplete finalization resumes on retry and
   does not report success until finalization is complete.
5. Crash recovery after the lifecycle hook boundary still leaves a resumable
   committed row.
6. Concurrent duplicate completions are idempotent.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated kanban home with an empty DB and no live-board leakage."""
    home = tmp_path / ".hermes"
    home.mkdir()
    kanban = home / "kanban"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(kanban))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban / "kanban.db"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
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


def test_canonical_digest_is_path_independent(kanban_home):
    """Identical bytes under different managed paths yield the same digest."""
    conn = kb.connect()
    t = kb.create_task(conn, title="path independent digest")
    task = kb.get_task(conn, t)
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, t, ws)

    first = ws / "a" / "report.txt"
    second = ws / "b" / "report.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("identical payload")
    second.write_text("identical payload")

    digest_one = kb._canonical_completion_digest(
        task_id=t,
        run_id=1,
        summary="done",
        result=None,
        metadata={"artifacts": [str(first)]},
        created_cards=None,
        review_pending=None,
    )
    digest_two = kb._canonical_completion_digest(
        task_id=t,
        run_id=1,
        summary="done",
        result=None,
        metadata={"artifacts": [str(second)]},
        created_cards=None,
        review_pending=None,
    )
    assert digest_one == digest_two
    assert digest_one.startswith("v1:")

    second.write_text("different payload")
    digest_three = kb._canonical_completion_digest(
        task_id=t,
        run_id=1,
        summary="done",
        result=None,
        metadata={"artifacts": [str(second)]},
        created_cards=None,
        review_pending=None,
    )
    assert digest_three != digest_one


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
    """If the DB transaction aborts, no durable completion row or attachment file is left."""
    conn = kb.connect()
    t = kb.create_task(conn, title="crash test")
    task = kb.get_task(conn, t)
    assert task is not None
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, t, ws)
    artifact = ws / "report.txt"
    artifact.write_text("findings")

    # Crash the main completion transaction before it commits by raising from
    # the run-ending helper. No ledger row or durable attachment may survive.
    original_end_run = kb._end_run

    def crashing_end_run(*args, **kwargs):
        raise RuntimeError("simulated crash before commit")

    monkeypatch.setattr(kb, "_end_run", crashing_end_run)
    with pytest.raises(RuntimeError, match="simulated crash before commit"):
        kb.complete_task(
            conn, t,
            summary="done",
            metadata={"artifacts": [str(artifact)]},
        )

    # The transaction rolled back: no completion ledger row, no attachments.
    row = conn.execute(
        "SELECT 1 FROM task_completion_results WHERE task_id = ?", (t,)
    ).fetchone()
    assert row is None
    assert _attachment_paths(conn, t) == []
    # Source scratch artifact is untouched because publication never ran.
    assert artifact.exists()


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

    # Restore the real finalizer and retry: now it finalizes.
    monkeypatch.setattr(kb, "_finalize_completion_result", original)
    result = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    assert result.ok
    row = conn.execute(
        "SELECT status FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "finalized"


def test_crash_during_finalization_resumes_and_cleans_workspace(populated_db, monkeypatch):
    """A real crash during finalization leaves a resumable committed row."""
    conn, tid, run_id, _child = populated_db
    task = kb.get_task(conn, tid)
    assert task is not None
    ws = kb.resolve_workspace(task)
    kb.set_workspace_path(conn, tid, ws)
    (ws / "report.txt").write_text("findings")

    calls = {"count": 0}
    real_recompute = kb.recompute_ready

    def wrapper(_conn):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("crash during recompute_ready")
        return real_recompute(_conn)

    monkeypatch.setattr(kb, "recompute_ready", wrapper)
    monkeypatch.setattr(kb, "_FINALIZATION_LEASE_SECONDS", 0)
    with pytest.raises(RuntimeError, match="crash during recompute_ready"):
        kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)

    # After the crash the ledger must still be committed (not finalized).
    row = conn.execute(
        "SELECT status FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] in {"committed", "finalizing"}
    # Workspace cleanup has not run yet.
    assert ws.exists()

    # Retry resumes finalization, finishes cleanup, and marks finalized.
    result = kb.complete_task(conn, tid, summary="done", expected_run_id=run_id)
    assert result.ok
    assert result.already_terminal is False

    row = conn.execute(
        "SELECT status, finalized_at FROM task_completion_results WHERE task_id=? AND run_id=?",
        (tid, run_id),
    ).fetchone()
    assert row["status"] == "finalized"
    assert row["finalized_at"] is not None
    assert not ws.exists()
    # First crash plus the successful retry means recompute ran twice total.
    assert calls["count"] == 2


def test_concurrent_duplicate_completion_is_idempotent(populated_db):
    """Two concurrent calls with the exact same payload produce exactly one transition."""
    conn, tid, run_id, _child = populated_db

    results: list[kb.CompletionResult] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def _complete():
        try:
            c = kb.connect()
            barrier.wait()
            results.append(
                kb.complete_task(c, tid, summary="done", expected_run_id=run_id)
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_complete) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r.ok for r in results)
    assert _task_status(conn, tid) == "done"
    terminal_run_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome IN ('completed', 'review_pending')",
            (tid,),
        ).fetchone()[0]
    )
    assert terminal_run_count == 1
    completed_events = int(
        conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
            (tid,),
        ).fetchone()[0]
    )
    assert completed_events == 1
    ledger_rows = int(
        conn.execute(
            "SELECT COUNT(*) FROM task_completion_results WHERE task_id=?",
            (tid,),
        ).fetchone()[0]
    )
    assert ledger_rows == 1
    assert (
        int(
            conn.execute(
                "SELECT COUNT(*) FROM task_completion_results WHERE task_id=? AND status IN ('committed', 'finalized')",
                (tid,),
            ).fetchone()[0]
        )
        == 1
    )


def test_synchronized_concurrent_duplicate_completion_exactly_once(kanban_home):
    """20-iteration synchronized race: exactly one terminal run/event/ledger each time."""
    failures: list[dict[str, Any]] = []
    for index in range(20):
        conn = kb.connect()
        task_id = kb.create_task(conn, title="probe", assignee="review-probe")
        assert kb.claim_task(conn, task_id) is not None
        run_id = int(
            conn.execute(
                "SELECT current_run_id FROM tasks WHERE id=?", (task_id,)
            ).fetchone()[0]
        )
        results: list[kb.CompletionResult] = []
        errors: list[str] = []
        barrier = threading.Barrier(2)

        def _complete():
            try:
                c = kb.connect()
                barrier.wait()
                results.append(
                    kb.complete_task(c, task_id, summary="done", expected_run_id=run_id)
                )
            except BaseException as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=_complete) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        counts: dict[str, Any] = {
            "results": len(results),
            "ok": sum(1 for r in results if r.ok),
            "errors": errors,
            "completed_runs": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='completed'",
                    (task_id,),
                ).fetchone()[0]
            ),
            "completed_events": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
                    (task_id,),
                ).fetchone()[0]
            ),
            "ledger_rows": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_completion_results WHERE task_id=?",
                    (task_id,),
                ).fetchone()[0]
            ),
        }
        expected = {
            "results": 2,
            "ok": 2,
            "errors": [],
            "completed_runs": 1,
            "completed_events": 1,
            "ledger_rows": 1,
        }
        if any(counts[key] != value for key, value in expected.items()):
            failures.append({"iteration": index, **counts})

    assert not failures, json.dumps(failures, indent=2)

def test_concurrent_duplicate_completion_return_semantics(kanban_home):
    """20-iteration race: winner gets already_terminal=False, loser gets True."""
    failures: list[dict[str, Any]] = []
    for index in range(20):
        conn = kb.connect()
        task_id = kb.create_task(conn, title="probe", assignee="review-probe")
        assert kb.claim_task(conn, task_id) is not None
        run_id = int(
            conn.execute(
                "SELECT current_run_id FROM tasks WHERE id=?", (task_id,)
            ).fetchone()[0]
        )
        results: list[kb.CompletionResult] = []
        errors: list[str] = []
        barrier = threading.Barrier(2)

        def _complete():
            try:
                c = kb.connect()
                barrier.wait()
                results.append(
                    kb.complete_task(c, task_id, summary="done", expected_run_id=run_id)
                )
            except BaseException as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=_complete) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        already_terminals = sorted(bool(r.already_terminal) for r in results)
        returned_run_ids = sorted(r.run_id for r in results)
        returned_summaries = sorted(r.summary for r in results)
        row = {
            "iteration": index,
            "results": len(results),
            "ok": sum(1 for r in results if r.ok),
            "already_terminal": already_terminals,
            "returned_run_ids": returned_run_ids,
            "returned_summaries": returned_summaries,
            "errors": errors,
            "completed_runs": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='completed'",
                    (task_id,),
                ).fetchone()[0]
            ),
            "completed_events": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
                    (task_id,),
                ).fetchone()[0]
            ),
            "ledger_rows": int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_completion_results WHERE task_id=?",
                    (task_id,),
                ).fetchone()[0]
            ),
        }
        expected = {
            "results": 2,
            "ok": 2,
            "already_terminal": [False, True],
            "returned_run_ids": [run_id, run_id],
            "returned_summaries": ["done", "done"],
            "errors": [],
            "completed_runs": 1,
            "completed_events": 1,
            "ledger_rows": 1,
        }
        mismatch = {
            key: {"expected": value, "actual": row[key]}
            for key, value in expected.items()
            if row[key] != value
        }
        if mismatch:
            failures.append({"iteration": index, "mismatch": mismatch})
        conn.close()

    assert not failures, json.dumps(failures, indent=2)


def test_historical_duplicate_completion_migration(kanban_home):
    """init_db repairs a legacy DB with duplicate (task_id, digest) rows.

    The unique index must not be created until dedupe has run; the survivor is
    the row with the lowest run_id (and lowest rowid as tie-breaker).
    """
    conn = kb.connect()
    conn.execute("DROP INDEX idx_completion_results_task_digest")
    insert_sql = """
        INSERT INTO task_completion_results (
            task_id, run_id, digest, status, terminal_status, outcome, summary, result,
            metadata_json, created_cards_json, review_pending_json, published_artifacts_json,
            terminal_event_id, completed_at, finalized_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def _ledger_values(run: int, summary: str) -> tuple:
        return (
            "historical-task", run, "same-digest", "committed", "done", "completed",
            summary, summary, "{}", "[]", "[]", "[]", None, 1000 + run, None,
        )

    conn.execute(insert_sql, _ledger_values(22, "later-row"))
    conn.execute(insert_sql, _ledger_values(11, "earlier-run"))
    conn.commit()
    conn.close()

    # Force reinit through the public entry point.
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
    db_path = Path(os.environ["HERMES_KANBAN_DB"])
    kb.init_db(db_path)

    raw = sqlite3.connect(str(db_path))
    rows = raw.execute(
        "SELECT run_id, summary FROM task_completion_results WHERE task_id=? ORDER BY run_id",
        ("historical-task",),
    ).fetchall()
    index_exists = bool(
        raw.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name='idx_completion_results_task_digest'"
        ).fetchone()[0]
    )
    raw.close()

    assert rows == [(11, "earlier-run")], rows
    assert index_exists

    # Second init is a no-op and remains consistent.
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
    kb.init_db(db_path)
    raw = sqlite3.connect(str(db_path))
    rows_after = raw.execute(
        "SELECT run_id, summary FROM task_completion_results WHERE task_id=? ORDER BY run_id",
        ("historical-task",),
    ).fetchall()
    raw.close()
    assert rows_after == [(11, "earlier-run")]


def test_synchronized_finalizer_hook_runs_exactly_once(kanban_home):
    """A racing loser must not run lifecycle-hook side effects."""
    conn = kb.connect()
    task_id = kb.create_task(conn, title="finalizer-race", assignee="review-probe")
    assert kb.claim_task(conn, task_id) is not None
    run_id = int(
        conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
    )

    hook_calls: list[dict[str, Any]] = []
    hook_lock = threading.Lock()

    def _counting_hook(*args, **kwargs):
        with hook_lock:
            hook_calls.append({"args": list(args), "kwargs": kwargs})

    original_hook = kb._fire_kanban_lifecycle_hook
    kb._fire_kanban_lifecycle_hook = _counting_hook
    try:
        barrier = threading.Barrier(2)
        results: list[kb.CompletionResult] = []
        errors: list[str] = []

        def _complete():
            try:
                c = kb.connect()
                barrier.wait()
                results.append(
                    kb.complete_task(c, task_id, summary="done", expected_run_id=run_id)
                )
            except BaseException as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=_complete) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert not errors
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert sorted(bool(r.already_terminal) for r in results) == [False, True]
        assert len(hook_calls) == 1
        assert (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed_hook_fired'",
                    (task_id,),
                ).fetchone()[0]
            )
            == 1
        )
        assert (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='completed'",
                    (task_id,),
                ).fetchone()[0]
            )
            == 1
        )
        assert (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
                    (task_id,),
                ).fetchone()[0]
            )
            == 1
        )
        assert (
            int(
                conn.execute(
                    "SELECT COUNT(*) FROM task_completion_results WHERE task_id=?",
                    (task_id,),
                ).fetchone()[0]
            )
            == 1
        )
    finally:
        kb._fire_kanban_lifecycle_hook = original_hook
    conn.close()
