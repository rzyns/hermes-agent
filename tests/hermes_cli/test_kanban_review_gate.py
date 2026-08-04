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
import time
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
        assert ok is True
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
        # Create a fake artifact in the scratch workspace.
        row = conn.execute(
            "SELECT workspace_path FROM tasks WHERE id = ?", (candidate,)
        ).fetchone()
        workspace = Path(row["workspace_path"])
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
        candidate_run_id = _run_id_for_task(conn, candidate)
        with pytest.raises(ValueError, match="cannot clear a review it requested"):
            kb.resolve_review(
                conn,
                candidate,
                "clear",
                resolving_run_id=candidate_run_id,
                reviewer_task_id=reviewer,
            )
        task = kb.get_task(conn, candidate)
        assert task.status == "review"
        assert kb.has_pending_review(conn, candidate) is True


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
        assert ok is True
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
        assert ok is True
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
        )
        assert ok is True
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

        # After resolving, a new request gets a new row.
        kb.resolve_review(
            conn,
            candidate,
            "reject",
            resolving_run_id=None,
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
    from tools import registry
    names = {t["function"]["name"] for t in registry.get_tool_definitions()}
    assert "kanban_resolve_review" in names


def test_kanban_create_schema_has_review_of_property():
    from tools import registry
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in registry.get_tool_definitions()}
    props = schemas.get("kanban_create", {}).get("properties", {})
    assert "review_of" in props


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
        # Simulate the candidate's own worker env.
        env = {
            "HERMES_KANBAN_TASK": candidate,
            "HERMES_KANBAN_RUN_ID": str(_run_id_for_task(conn, candidate)),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            result = _handle_resolve_review(
                {"candidate_id": candidate, "resolution": "clear"}
            )
        data = _json_parse(result)
        assert data.get("ok") is False
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
