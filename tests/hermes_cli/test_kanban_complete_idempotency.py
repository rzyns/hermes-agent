"""Tests for idempotent, crash-safe kanban completion.

Covers:
- Condition A: canonical digest of completion inputs.
- Condition B: staged outbox for artifact publication / finalization.
- Durable completion result keyed by (task_id, run_id) + digest.
- Idempotent post-commit finalization and orphan staging cleanup.
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


def _task_status(conn: sqlite3.Connection, task_id: str) -> str:
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row is not None
    return row["status"]


def _task_run_count(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?", (task_id,)
    ).fetchone()
    return int(row["n"])


def _completed_events(conn: sqlite3.Connection, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'completed' ORDER BY id",
        (task_id,),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def _attachment_paths(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT stored_path FROM task_attachments WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [r["stored_path"] for r in rows]


# ---------------------------------------------------------------------------
# Basic idempotency
# ---------------------------------------------------------------------------


def test_duplicate_complete_with_same_run_and_payload_is_idempotent(populated_db):
    conn, tid, run_id, _child = populated_db
    assert kb.complete_task(
        conn, tid,
        summary="first completion",
        expected_run_id=run_id,
    )
    assert _task_status(conn, tid) == "done"
    first_run_count = _task_run_count(conn, tid)
    first_events = _completed_events(conn, tid)

    # Exact retry must succeed without creating a new run or event.
    assert kb.complete_task(
        conn, tid,
        summary="first completion",
        expected_run_id=run_id,
    )
    assert _task_status(conn, tid) == "done"
    assert _task_run_count(conn, tid) == first_run_count
    assert _completed_events(conn, tid) == first_events


def test_duplicate_complete_with_different_digest_raises_conflict(populated_db):
    conn, tid, run_id, _child = populated_db
    assert kb.complete_task(
        conn, tid,
        summary="first completion",
        expected_run_id=run_id,
    )

    with pytest.raises(kb.CompletionDigestConflictError) as excinfo:
        kb.complete_task(
            conn, tid,
            summary="changed summary",
            expected_run_id=run_id,
        )
    assert excinfo.value.task_id == tid
    assert excinfo.value.run_id == run_id
    assert _task_status(conn, tid) == "done"


def test_blank_result_still_raises_before_any_idempotency_record(populated_db):
    conn, tid, run_id, _child = populated_db
    with pytest.raises(kb.CompletionResultRequiredError, match=str(tid)):
        kb.complete_task(conn, tid, summary="", expected_run_id=run_id)

    # No durable completion record should have been written.
    row = conn.execute(
        "SELECT 1 FROM task_completion_results WHERE task_id = ? AND run_id = ?",
        (tid, run_id),
    ).fetchone()
    assert row is None


def test_phantom_created_cards_still_raises_and_does_not_write_digest(populated_db):
    conn, tid, run_id, _child = populated_db
    phantom_id = "t_deadbeefcafe"
    with pytest.raises(kb.HallucinatedCardsError) as excinfo:
        kb.complete_task(
            conn, tid,
            summary="done",
            created_cards=[phantom_id],
            expected_run_id=run_id,
        )
    assert excinfo.value.phantom == [phantom_id]

    row = conn.execute(
        "SELECT 1 FROM task_completion_results WHERE task_id = ? AND run_id = ?",
        (tid, run_id),
    ).fetchone()
    assert row is None


# ---------------------------------------------------------------------------
# Digest canonicalization
# ---------------------------------------------------------------------------


def test_digest_treats_summary_and_result_as_equivalent_when_same_effective(populated_db):
    conn, tid, run_id, _child = populated_db
    # First call uses summary.
    assert kb.complete_task(conn, tid, summary="effective", expected_run_id=run_id)
    # Retry with equivalent effective result should be idempotent.
    assert kb.complete_task(conn, tid, result="effective", expected_run_id=run_id)
    assert _task_status(conn, tid) == "done"
    assert _task_run_count(conn, tid) == 1


def test_digest_normalizes_whitespace_in_summary(populated_db):
    conn, tid, run_id, _child = populated_db
    assert kb.complete_task(
        conn, tid,
        summary="hello world",
        expected_run_id=run_id,
    )
    assert kb.complete_task(
        conn, tid,
        summary="  hello world  ",
        expected_run_id=run_id,
    )
    assert _task_run_count(conn, tid) == 1


def test_digest_includes_metadata_and_created_cards(populated_db):
    conn, tid, run_id, child = populated_db
    assert kb.complete_task(
        conn, tid,
        summary="done",
        metadata={"changed_files": ["a.py"]},
        created_cards=[child],
        expected_run_id=run_id,
    )
    # Same summary but different metadata -> conflict.
    with pytest.raises(kb.CompletionDigestConflictError):
        kb.complete_task(
            conn, tid,
            summary="done",
            metadata={"changed_files": ["b.py"]},
            created_cards=[child],
            expected_run_id=run_id,
        )
    # Same summary/metadata but different created_cards -> conflict.
    other = kb.create_task(conn, title="other", assignee="x", created_by="alice")
    with pytest.raises(kb.CompletionDigestConflictError):
        kb.complete_task(
            conn, tid,
            summary="done",
            metadata={"changed_files": ["a.py"]},
            created_cards=[child, other],
            expected_run_id=run_id,
        )


# ---------------------------------------------------------------------------
# Artifact staging / finalization
# ---------------------------------------------------------------------------


def test_duplicate_complete_does_not_duplicate_scratch_artifacts(populated_db, kanban_home):
    conn, tid, run_id, _child = populated_db
    task = kb.get_task(conn, tid)
    assert task is not None
    workspace = kb.resolve_workspace(task)
    workspace.mkdir(parents=True, exist_ok=True)
    artifact = workspace / "report.txt"
    artifact.write_text("findings", encoding="utf-8")

    assert kb.complete_task(
        conn, tid,
        summary="done",
        metadata={"artifacts": [str(artifact)]},
        expected_run_id=run_id,
    )
    first_paths = _attachment_paths(conn, tid)
    assert len(first_paths) == 1
    assert Path(first_paths[0]).exists()

    # Retry with the same digest must not add another attachment row or file copy.
    assert kb.complete_task(
        conn, tid,
        summary="done",
        metadata={"artifacts": [str(artifact)]},
        expected_run_id=run_id,
    )
    second_paths = _attachment_paths(conn, tid)
    assert second_paths == first_paths
    assert _task_run_count(conn, tid) == 1


def test_orphan_staged_artifacts_are_finalized_on_later_idempotent_retry(populated_db, kanban_home):
    """Simulate a crash after DB commit but before filesystem finalization.

    The staged file was already copied but the attachment row was not
    recorded. A later retry with the same digest must record the row and
    not duplicate the file.
    """
    conn, tid, run_id, _child = populated_db
    task = kb.get_task(conn, tid)
    assert task is not None
    workspace = kb.resolve_workspace(task)
    workspace.mkdir(parents=True, exist_ok=True)
    artifact = workspace / "report.txt"
    artifact.write_text("findings", encoding="utf-8")

    # First, capture the staged path by calling the staging helper directly.
    # Compute the digest from the *input* metadata (before staging rewrites
    # artifact paths), matching the canonicalization used by complete_task.
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
    metadata = dict(input_metadata)
    staged = kb._persist_scratch_completion_artifacts(conn, tid, metadata)
    assert len(staged) == 1
    staged_path = staged[0]
    assert staged_path.exists()

    # Write the durable completion record as if the DB transaction committed
    # but the attachment-row write crashed.
    conn.execute(
        "INSERT INTO task_completion_results "
        "(task_id, run_id, digest, status, terminal_status, outcome, "
        " summary, result, metadata_json, created_cards_json, "
        " published_artifacts_json, completed_at) "
        "VALUES (?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            tid,
            run_id,
            digest,
            "done",
            "completed",
            "done",
            None,
            json.dumps(metadata, ensure_ascii=False),
            json.dumps([], ensure_ascii=False),
            json.dumps([str(staged_path)], ensure_ascii=False),
            int(time.time()),
        ),
    )
    conn.commit()

    # Retry should finalize the orphan attachment row without duplicating files.
    assert kb.complete_task(
        conn, tid,
        summary="done",
        metadata=input_metadata,
        expected_run_id=run_id,
    )
    paths = _attachment_paths(conn, tid)
    assert paths == [str(staged_path)]
    assert _task_run_count(conn, tid) == 1


# ---------------------------------------------------------------------------
# Preserved existing behavior
# ---------------------------------------------------------------------------


def test_expected_run_id_still_enforced_for_unknown_run(populated_db):
    conn, tid, _run_id, _child = populated_db
    assert kb.complete_task(
        conn, tid,
        summary="done",
        expected_run_id=999999,
    ).ok is False


def test_budget_bench_grace_path_still_works(kanban_home, monkeypatch):
    # The grace config lives in config.yaml under kanban.terminal_completion_grace_seconds.
    # Patch the helper directly so the test does not depend on config loader internals.
    import hermes_cli.kanban_db as _kb
    monkeypatch.setattr(_kb, "_terminal_completion_grace_seconds", lambda: 60)
    conn = kb.connect()
    tid = kb.create_task(conn, title="budget task", assignee="alice")
    kb.claim_task(conn, tid)
    run_id = _current_run_id(conn, tid)

    # Simulate a budget-exhausted run ending via the failure path (the same
    # path the agent loop uses when it runs out of iterations). Manual
    # block_task(kind='budget') creates a 'blocked' outcome that is not part
    # of the terminal grace contract.
    kb._record_task_failure(
        conn, tid,
        error="Iteration budget exhausted (4/4)",
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        expected_run_id=run_id,
        failure_limit=1,
    )
    assert _task_status(conn, tid) == "blocked"

    # Completion within grace should succeed and not be treated as duplicate.
    assert kb.complete_task(conn, tid, summary="recovered", expected_run_id=run_id)
    assert _task_status(conn, tid) == "done"


def test_review_pending_idempotent_across_retries(populated_db):
    conn, tid, run_id, _child = populated_db
    reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
    assert kb.complete_task(
        conn, tid,
        summary="needs review",
        review_pending=[reviewer],
        expected_run_id=run_id,
    )
    first_status = _task_status(conn, tid)
    first_request_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM review_requests WHERE candidate_id = ?",
            (tid,),
        ).fetchone()[0]
    )

    assert kb.complete_task(
        conn, tid,
        summary="needs review",
        review_pending=[reviewer],
        expected_run_id=run_id,
    )
    assert _task_status(conn, tid) == first_status
    assert int(
        conn.execute(
            "SELECT COUNT(*) FROM review_requests WHERE candidate_id = ?",
            (tid,),
        ).fetchone()[0]
    ) == first_request_count


# ---------------------------------------------------------------------------
# Schema migration coverage
# ---------------------------------------------------------------------------


def test_legacy_db_without_completion_results_table_migrates_cleanly(tmp_path, monkeypatch):
    """A board DB created before the completion-result table still opens."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Create the legacy schema without the new table.
    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(kb.SCHEMA_SQL)
    legacy.commit()
    legacy.close()

    # init_db must add the new table/indexes.
    kb.init_db()
    with kb.connect() as conn:
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "task_completion_results" in tables
        indexes = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "idx_completion_results_task_run" in indexes
        assert "idx_completion_results_digest" in indexes
