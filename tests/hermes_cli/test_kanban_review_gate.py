"""Deterministic tests for the independent review gate.

This test file verifies the semantics of ``kanban_complete(review_pending=...)``
and ``kanban_resolve_review``:

- review-pending is a handoff, not a block
- self-approval is rejected
- chronology is preserved across rejection/re-review rounds
- dependent tasks wait while a done parent still has pending review
- reviewer tasks are runnable independently
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any
from unittest import mock

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


# ---------------------------------------------------------------------------
# DB-layer review gate
# ---------------------------------------------------------------------------


def _run_id_for_task(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    assert row is not None and row["current_run_id"] is not None
    return int(row["current_run_id"])


def _last_run_id_for_task(conn: sqlite3.Connection, task_id: str) -> int:
    row = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? ORDER BY ended_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def test_complete_with_review_pending_moves_to_review_not_done(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        ok = kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        assert ok
        task = kb.get_task(conn, candidate)
        assert task.status == "review"
        assert task.completed_at is None
        run = kb.latest_run(conn, candidate)
        assert run.outcome == "review_pending"


def test_complete_with_review_pending_creates_pending_request(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        requests = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(requests) == 1
        assert requests[0].status == "pending"
        assert requests[0].reviewer_task_id == reviewer
        assert requests[0].requested_by_run_id is not None


def test_review_pending_preserves_scratch_workspace(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        # Resolve the candidate's scratch workspace.
        workspace = kb.resolve_workspace(kb.get_task(conn, candidate))
        workspace.mkdir(parents=True, exist_ok=True)
        artifact = workspace / "result.txt"
        artifact.write_text("candidate output", encoding="utf-8")

        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        assert artifact.exists(), "scratch workspace must be preserved for repair"


def test_self_relation_is_rejected_at_complete_and_request(kanban_home):
    with kb.connect() as conn:
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        # complete_task with the candidate as its own reviewer must fail.
        with pytest.raises(ValueError, match="cannot review itself"):
            kb.complete_task(
                conn,
                candidate,
                summary="finished",
                review_pending=[candidate],
            )
        assert kb.get_task(conn, candidate).status == "running"
        # request_review with candidate==reviewer must also fail.
        with pytest.raises(ValueError, match="cannot review itself"):
            kb.request_review(conn, candidate, candidate, requested_by_run_id=None)


def test_resolve_review_requires_run_and_operator_clear_review_works(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        # A worker tool call with no resolving run id is rejected.
        with pytest.raises(ValueError, match="resolving run id is required"):
            kb.resolve_review(
                conn,
                candidate,
                "clear",
                resolving_run_id=None,
                reviewer_task_id=reviewer,
            )
        # The worker path no longer accepts an operator_mode boolean.
        with pytest.raises(TypeError):
            kb.resolve_review(
                conn,
                candidate,
                "clear",
                resolving_run_id=None,
                reviewer_task_id=reviewer,
                operator_mode=True,
            )
        # Operator path requires an attributable actor.
        with pytest.raises(ValueError, match="operator_actor"):
            kb.operator_clear_review(
                conn,
                candidate,
                "clear",
                "",
            )
        ok = kb.operator_clear_review(
            conn,
            candidate,
            "clear",
            "cli:test-operator",
            reviewer_task_id=reviewer,
        )
        assert ok
        assert kb.get_task(conn, candidate).status == "done"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_cleared'",
            (candidate,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload.get("operator_actor") == "cli:test-operator"
        requests = kb.get_review_requests(conn, candidate_id=candidate)
        assert requests[-1].resolved_by_operator == "cli:test-operator"
        assert requests[-1].resolved_by_run_id is None


def test_multiple_reviewers_clear_and_gate(kanban_home):
    with kb.connect() as conn:
        r1 = kb.create_task(conn, title="r1", assignee="reviewer")
        r2 = kb.create_task(conn, title="r2", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished",
            review_pending=[r1, r2],
        )
        kb.claim_task(conn, r1)
        rid1 = _run_id_for_task(conn, r1)
        kb.resolve_review(conn, candidate, "clear", resolving_run_id=rid1, reviewer_task_id=r1)
        # One clearance is not enough.
        assert kb.get_task(conn, candidate).status == "review"
        assert kb.has_pending_review(conn, candidate) is True
        # Second clearance completes the candidate.
        kb.claim_task(conn, r2)
        rid2 = _run_id_for_task(conn, r2)
        kb.resolve_review(conn, candidate, "clear", resolving_run_id=rid2, reviewer_task_id=r2)
        assert kb.get_task(conn, candidate).status == "done"


def test_reject_vetoes_all_pending_reviewers(kanban_home):
    with kb.connect() as conn:
        r1 = kb.create_task(conn, title="r1", assignee="reviewer")
        r2 = kb.create_task(conn, title="r2", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished",
            review_pending=[r1, r2],
        )
        kb.claim_task(conn, r1)
        rid1 = _run_id_for_task(conn, r1)
        kb.resolve_review(conn, candidate, "reject", resolving_run_id=rid1, reviewer_task_id=r1)
        # Candidate is back to ready; all requests are rejected.
        assert kb.get_task(conn, candidate).status == "ready"
        reqs = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(reqs) == 2
        assert all(r.status == "rejected" for r in reqs)


def test_self_approval_by_candidate_run_is_rejected(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        candidate_run_id = _last_run_id_for_task(conn, candidate)
        # The candidate run is not associated with the reviewer task, so
        # resolving with that run id is rejected as not belonging to the
        # reviewer. This is the same practical outcome: a candidate cannot
        # clear its own review.
        with pytest.raises(ValueError):
            kb.resolve_review(
                conn,
                candidate,
                resolution="clear",
                resolving_run_id=candidate_run_id,
                reviewer_task_id=reviewer,
            )
        task = kb.get_task(conn, candidate)
        assert task.status == "review"
        assert kb.has_pending_review(conn, candidate) is True


def test_resolve_review_rejects_stale_ended_reviewer_run(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        stale_run_id = _run_id_for_task(conn, reviewer)
        # Reclaim the reviewer task, clearing current_run_id and claim_lock.
        kb.reclaim_task(conn, reviewer)
        assert kb.get_task(conn, reviewer).current_run_id is None
        assert kb.get_task(conn, reviewer).claim_lock is None
        assert kb.get_task(conn, reviewer).status == "ready"

        # Stale run can no longer clear the review.
        with pytest.raises(ValueError, match="not the current run"):
            kb.resolve_review(
                conn,
                candidate,
                "clear",
                resolving_run_id=stale_run_id,
                reviewer_task_id=reviewer,
            )
        task = kb.get_task(conn, candidate)
        assert task.status == "review"
        assert kb.has_pending_review(conn, candidate) is True


def test_tool_resolve_review_rejects_missing_claim_lock(kanban_home):
    from tools.kanban_tools import _handle_resolve_review

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        env = {
            "HERMES_KANBAN_TASK": reviewer,
            "HERMES_KANBAN_RUN_ID": str(_run_id_for_task(conn, reviewer)),
            # Deliberately omit HERMES_KANBAN_CLAIM_LOCK.
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert "error" in data
        assert "claim lock" in data["error"].lower()
        assert kb.get_task(conn, candidate).status == "review"


def test_tool_resolve_review_rejects_stale_reviewer_run(kanban_home):
    from tools.kanban_tools import _handle_resolve_review

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        stale_run_id = _run_id_for_task(conn, reviewer)
        stale_lock = kb.get_task(conn, reviewer).claim_lock
        kb._end_run(conn, reviewer, outcome="timed_out")

        env = {
            "HERMES_KANBAN_TASK": reviewer,
            "HERMES_KANBAN_RUN_ID": str(stale_run_id),
            "HERMES_KANBAN_CLAIM_LOCK": str(stale_lock) if stale_lock else "stale",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert "error" in data
        assert kb.get_task(conn, candidate).status == "review"


def test_bound_reviewer_can_clear_review(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        reviewer_run_id = _run_id_for_task(conn, reviewer)
        ok = kb.resolve_review(
            conn,
            candidate,
            "clear",
            resolving_run_id=reviewer_run_id,
            reviewer_task_id=reviewer,
        )
        assert ok
        task = kb.get_task(conn, candidate)
        assert task.status == "done"
        assert task.completed_at is not None
        assert kb.has_pending_review(conn, candidate) is False


def test_reject_review_returns_candidate_to_ready(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        reviewer_run_id = _run_id_for_task(conn, reviewer)
        ok = kb.resolve_review(
            conn,
            candidate,
            "reject",
            resolving_run_id=reviewer_run_id,
            reviewer_task_id=reviewer,
            reason="needs more tests",
        )
        assert ok
        task = kb.get_task(conn, candidate)
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.claim_lock is None


def test_reject_review_returns_candidate_to_todo_when_parents_unfinished(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(
            conn, title="candidate", assignee="worker", parents=[parent]
        )
        # Claim + complete parent so candidate becomes ready before review.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="parent done")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        # Roll parent back to running is impossible; instead we fake an
        # unfinished parent by updating its status directly after the
        # candidate is already in review, so rejection recovers to todo.
        conn.execute("UPDATE tasks SET status='running' WHERE id = ?", (parent,))
        kb.claim_task(conn, reviewer)
        reviewer_run_id = _run_id_for_task(conn, reviewer)
        ok = kb.resolve_review(
            conn,
            candidate,
            "reject",
            resolving_run_id=reviewer_run_id,
            reviewer_task_id=reviewer,
        )
        assert ok
        task = kb.get_task(conn, candidate)
        assert task.status == "todo"


def test_review_chronology_preserved_across_rounds(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")

        # First round: complete, reject.
        kb.claim_task(conn, candidate)
        kb.complete_task(conn, candidate, summary="v1", review_pending=[reviewer])
        kb.claim_task(conn, reviewer)
        rid1 = _run_id_for_task(conn, reviewer)
        kb.resolve_review(
            conn, candidate, "reject", resolving_run_id=rid1, reviewer_task_id=reviewer
        )
        first_requests = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(first_requests) == 1
        assert first_requests[0].status == "rejected"

        # Second round: re-complete, clear.
        kb.claim_task(conn, candidate)
        kb.complete_task(conn, candidate, summary="v2", review_pending=[reviewer])
        kb.claim_task(conn, reviewer)
        rid2 = _run_id_for_task(conn, reviewer)
        kb.resolve_review(
            conn, candidate, "clear", resolving_run_id=rid2, reviewer_task_id=reviewer
        )
        all_requests = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(all_requests) == 2
        assert all_requests[0].status == "rejected"
        assert all_requests[1].status == "cleared"
        assert all_requests[1].id > all_requests[0].id


def test_request_review_is_idempotent_for_same_pending_pair(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        # Make candidate review-bound so request_review accepts it directly.
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn, candidate, summary="v1", review_pending=[reviewer]
        )
        run_id_1 = 12345
        rid1 = kb.request_review(
            conn, candidate, reviewer, requested_by_run_id=run_id_1, reason="first"
        )
        rid2 = kb.request_review(
            conn, candidate, reviewer, requested_by_run_id=run_id_1, reason="second"
        )
        assert rid1 == rid2
        pending = kb.get_review_requests(
            conn, candidate_id=candidate, status="pending"
        )
        assert len(pending) == 1
        assert pending[0].reason == "second"

        # After resolving, a new request in the same second must create a
        # distinct row because the primary identity is the row id / random
        # idempotency key, not the wall clock.
        kb.operator_clear_review(
            conn,
            candidate,
            "reject",
            "test:fixture",
            reviewer_task_id=reviewer,
        )
        rid3 = kb.request_review(
            conn, candidate, reviewer, requested_by_run_id=run_id_1, reason="third"
        )
        assert rid3 != rid1
        all_reqs = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(all_reqs) == 2


def test_review_status_does_not_increment_block_recurrences_or_triage(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        task = kb.get_task(conn, candidate)
        assert task.status == "review"
        assert task.block_recurrences == 0
        assert task.block_kind is None


def test_recompute_ready_does_not_promote_children_of_done_parent_pending_review(
    kanban_home,
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent]
        )
        # Parent completes but needs review.
        kb.claim_task(conn, parent)
        kb.complete_task(
            conn, parent, summary="parent done but under review", review_pending=[reviewer]
        )
        assert kb.get_task(conn, parent).status == "review"
        # Child must stay todo: parent is "done" but has pending review.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"


def test_child_promoted_after_review_cleared(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        child = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent]
        )
        kb.claim_task(conn, parent)
        kb.complete_task(
            conn, parent, summary="parent done but under review", review_pending=[reviewer]
        )
        kb.claim_task(conn, reviewer)
        rid = _run_id_for_task(conn, reviewer)
        kb.resolve_review(conn, parent, "clear", resolving_run_id=rid, reviewer_task_id=reviewer)
        assert kb.get_task(conn, parent).status == "done"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_reviewer_task_is_independently_runnable(kanban_home, all_assignees_spawnable):
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    # Reviewer is ready and independent; it should be spawned.
    assert reviewer in spawns or reviewer in [s[0] for s in res.spawned]
    # Candidate is in review and must NOT be re-spawned as ready.
    assert candidate not in spawns


def test_no_task_links_between_candidate_and_reviewer(kanban_home):
    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        rows = conn.execute(
            "SELECT * FROM task_links "
            "WHERE (parent_id = ? AND child_id = ?) "
            "   OR (parent_id = ? AND child_id = ?)",
            (candidate, reviewer, reviewer, candidate),
        ).fetchall()
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# Tool-layer review gate
# ---------------------------------------------------------------------------


def test_kanban_resolve_review_schema_registered():
    # Import the tool file so its module-level registry.register() calls run.
    import tools.kanban_tools  # noqa: F401
    from tools import registry
    names = {entry.name for entry in registry.registry._snapshot_entries()}
    assert "kanban_resolve_review" in names


def test_schema_migration_removes_old_second_resolution_unique_key(kanban_home):
    """Legacy boards with the old 4-column UNIQUE key upgrade cleanly."""
    db_path = kanban_home / "kanban.db"
    # The fixture already initialized the DB. Drop the modern table and
    # recreate the legacy schema to simulate a board from the first deploy.
    raw = sqlite3.connect(str(db_path))
    raw.execute("DROP TABLE IF EXISTS review_requests")
    raw.execute(
        """
        CREATE TABLE review_requests (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id       TEXT NOT NULL,
            reviewer_task_id   TEXT NOT NULL,
            status             TEXT NOT NULL DEFAULT 'pending',
            requested_by_run_id INTEGER,
            requested_at       INTEGER NOT NULL,
            resolved_at        INTEGER,
            resolved_by_run_id INTEGER,
            resolution         TEXT,
            reason             TEXT,
            UNIQUE(candidate_id, reviewer_task_id, requested_at, requested_by_run_id)
        )
        """
    )
    raw.execute(
        "INSERT INTO review_requests (candidate_id, reviewer_task_id, status, requested_at) "
        "VALUES ('cand', 'rev', 'pending', 1)"
    )
    raw.commit()
    raw.close()

    kb.init_db()
    with kb.connect() as conn:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(review_requests)")
        }
        assert "idempotency_key" in cols
        assert "resolved_by_operator" in cols
        # Old 4-column unique key should be gone. A new autoindex from
        # UNIQUE(idempotency_key) is expected, so we inspect column sets.
        autoindexes = conn.execute(
            """
            SELECT name FROM sqlite_master
             WHERE type = 'index'
               AND tbl_name = 'review_requests'
               AND sql IS NULL
               AND name LIKE 'sqlite_autoindex_review_requests_%'
            """
        ).fetchall()
        for idx in autoindexes:
            idx_info = conn.execute(f"PRAGMA index_info({idx['name']!r})").fetchall()
            idx_cols = tuple(row["name"] for row in idx_info)
            assert idx_cols != (
                "candidate_id",
                "reviewer_task_id",
                "requested_at",
                "requested_by_run_id",
            )
        rows = conn.execute("SELECT * FROM review_requests").fetchall()
        assert len(rows) == 1
        assert rows[0]["candidate_id"] == "cand"
        assert rows[0]["idempotency_key"].startswith("legacy:")


def test_tool_resolve_review_requires_worker_env(kanban_home):
    from tools.kanban_tools import _handle_resolve_review

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert "error" in data
        assert "HERMES_KANBAN_TASK" in data["error"]


def test_tool_resolve_review_rejects_self_approval(kanban_home):
    from tools.kanban_tools import _handle_resolve_review

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        # Simulate the candidate's own worker env using the just-ended run.
        env = {
            "HERMES_KANBAN_TASK": candidate,
            "HERMES_KANBAN_RUN_ID": str(_last_run_id_for_task(conn, candidate)),
        }
        with mock.patch.dict(os.environ, env, clear=True):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert data.get("ok") is False or "error" in data
        task = kb.get_task(conn, candidate)
        assert task.status == "review"


def test_tool_resolve_review_bound_reviewer_can_clear(kanban_home):
    from tools.kanban_tools import _handle_resolve_review

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )
        kb.claim_task(conn, reviewer)
        env = {
            "HERMES_KANBAN_TASK": reviewer,
            "HERMES_KANBAN_RUN_ID": str(_run_id_for_task(conn, reviewer)),
            "HERMES_KANBAN_CLAIM_LOCK": kb.get_task(conn, reviewer).claim_lock,
        }
        with mock.patch.dict(os.environ, env, clear=False):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert data.get("ok") is True
        assert data.get("status") == "done"


def test_tool_create_with_review_of(kanban_home):
    from tools.kanban_tools import _handle_create

    with kb.connect() as conn:
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        candidate_run_id = _run_id_for_task(conn, candidate)
        env = {
            "HERMES_KANBAN_TASK": candidate,
            "HERMES_KANBAN_RUN_ID": str(candidate_run_id),
            "HERMES_PROFILE": "worker",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            result = _handle_create(
                {
                    "title": "reviewer",
                    "assignee": "reviewer",
                    "review_of": candidate,
                }
            )
        data = _json_parse(result)
        assert data.get("ok") is True
        reviewer_id = data["task_id"]
        requests = kb.get_review_requests(conn, candidate_id=candidate)
        assert len(requests) == 1
        assert requests[0].reviewer_task_id == reviewer_id


def test_tool_complete_with_review_pending(kanban_home):
    from tools.kanban_tools import _handle_complete

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        env = {
            "HERMES_KANBAN_TASK": candidate,
            "HERMES_KANBAN_RUN_ID": str(_run_id_for_task(conn, candidate)),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            result = _handle_complete(
                {
                    "summary": "finished",
                    "review_pending": [reviewer],
                }
            )
        data = _json_parse(result)
        assert data.get("ok") is True
        task = kb.get_task(conn, candidate)
        assert task.status == "review"


def _json_parse(text: str) -> dict[str, Any]:
    import json
    return json.loads(text)


import json  # noqa: E402


def test_cli_review_clear_as_operator(kanban_home, capsys):
    """hermes kanban review-clear records the operator actor durably."""
    import argparse
    import hermes_cli.kanban as kanban_cli

    with kb.connect() as conn:
        reviewer = kb.create_task(conn, title="reviewer", assignee="reviewer")
        candidate = kb.create_task(conn, title="candidate", assignee="worker")
        kb.claim_task(conn, candidate)
        kb.complete_task(
            conn,
            candidate,
            summary="finished; needs review",
            review_pending=[reviewer],
        )

    parent = argparse.ArgumentParser().add_subparsers()
    parser = kanban_cli.build_parser(parent_subparsers=parent)
    args = parser.parse_args(
        ["review-clear", candidate, "--operator", "ops:alice", "--json"]
    )
    rc = kanban_cli._cmd_review_clear(args)
    assert rc == 0
    out, err = capsys.readouterr()
    data = json.loads(out)
    assert data["ok"] is True
    assert data["status"] == "done"
    assert data["operator_actor"] == "cli:ops:alice"

    with kb.connect() as conn:
        req = kb.get_review_requests(conn, candidate_id=candidate)[0]
        assert req.resolved_by_operator == "cli:ops:alice"
        assert req.resolved_by_run_id is None
        assert req.resolution == "clear"
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_cleared'",
            (candidate,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert payload.get("operator_actor") == "cli:ops:alice"
