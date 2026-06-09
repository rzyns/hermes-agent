"""Tests for the kanban-cross-deps dependency provider through the core seam.

Uses temp HERMES_HOME so no live board state is touched.
These exercises scheduler invariants: recompute_ready and claim_task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_dependencies import ExternalBlocker, scoped_dependency_provider
from plugins.kanban_cross_deps.provider import CrossBoardDependencyProvider
from plugins.kanban_cross_deps.store import CrossBoardRegistry


def _assert_blocker(b):
    assert isinstance(b, ExternalBlocker)
    return b


@pytest.fixture
def kcd_home(tmp_path, monkeypatch):
    """Temp Hermes home with initialized kanban DB for both parent and child boards."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Prevent leaked kanban overrides from affecting board/registry resolution
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    # Assert that the board DB resolved under the temp home
    assert kb.kanban_db_path().resolve().is_relative_to(home.resolve())
    return home


def _create_board(board: str, home: Path):
    """Ensure a board DB exists and is initialized."""
    db_path = kb.kanban_db_path(board=board)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = kb.connect(db_path=db_path, board=board)
    conn.close()


# ---------------------------------------------------------------------------
# Provider unit tests
# ---------------------------------------------------------------------------

class TestProviderUnit:
    def test_returns_unsatisfied_for_todo_parent(self, kcd_home):
        reg = CrossBoardRegistry()
        # Parent task on parent-board in todo
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
        finally:
            p_conn.close()

        # Child edge
        edge = reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id="t_child",
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        blockers = list(provider.blockers_for(board="default", task_id="t_child"))
        assert len(blockers) == 1
        b = _assert_blocker(blockers[0])
        assert b.satisfied is False
        assert b.status in ("todo", "ready")
        assert b.parent_board == "parent-board"
        assert b.parent_id == pid

    def test_returns_none_for_done_parent(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id="t_child",
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        blockers = list(provider.blockers_for(board="default", task_id="t_child"))
        # Satisfied edge → provider should not return it (only unsatisfied blockers)
        assert blockers == []

    def test_ignores_non_blocking_edges(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
        finally:
            p_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id="t_child",
            kind="informed_by", blocking=False,
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        blockers = list(provider.blockers_for(board="default", task_id="t_child"))
        assert blockers == []

    def test_fail_closed_for_dangling_parent(self, kcd_home):
        reg = CrossBoardRegistry()
        reg.add(
            parent_board="missing-board", parent_id="t_missing",
            child_board="default", child_id="t_child",
            kind="blocks",
        )
        provider = CrossBoardDependencyProvider(registry=reg)
        blockers = list(provider.blockers_for(board="default", task_id="t_child"))
        assert len(blockers) == 1
        b = _assert_blocker(blockers[0])
        assert b.satisfied is False
        assert "not found" in b.reason or "missing-board" in b.reason

    def test_respects_required_parent_statuses(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
            with kb.write_txn(p_conn):
                p_conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (pid,))
        finally:
            p_conn.close()

        # Edge that requires only "todo" to be satisfied (unusual but tests param)
        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id="t_child",
            kind="blocks",
            required_parent_statuses=["todo"],
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        blockers = list(provider.blockers_for(board="default", task_id="t_child"))
        assert blockers == []  # satisfied because parent is todo


# ---------------------------------------------------------------------------
# Core seam integration: recompute_ready
# ---------------------------------------------------------------------------

class TestProviderBlocksRecomputeReady:
    def test_unsatisfied_blocks_promotion(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
        finally:
            p_conn.close()

        # Child on default board with no local parents; force todo so
        # recompute_ready processes it rather than skipping a ready task.
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                promoted = kb.recompute_ready(c_conn, board="default")
            assert promoted == 0
            task = kb.get_task(c_conn, cid)
            assert task.status == "todo"
        finally:
            c_conn.close()

    def test_satisfied_allows_promotion(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                promoted = kb.recompute_ready(c_conn, board="default")
            assert promoted == 1
            task = kb.get_task(c_conn, cid)
            assert task.status == "ready"
        finally:
            c_conn.close()


# ---------------------------------------------------------------------------
# Core seam integration: claim_task
# ---------------------------------------------------------------------------

class TestProviderBlocksClaimTask:
    def test_unsatisfied_blocks_claim_and_demotes(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child", assignee="worker")
            # Manually promote to ready so we can test claim rejection
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                claimed = kb.claim_task(c_conn, cid, board="default")
            assert claimed is None
            task = kb.get_task(c_conn, cid)
            assert task.status == "todo"
            events = kb.list_events(c_conn, cid)
            rejected = [e for e in events if e.kind == "claim_rejected"]
            assert rejected
            assert rejected[-1].payload["reason"] == "external_blockers"
        finally:
            c_conn.close()

    def test_satisfied_allows_claim(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child", assignee="worker")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                claimed = kb.claim_task(c_conn, cid, board="default")
            assert claimed is not None
            assert claimed.status == "running"
        finally:
            c_conn.close()

    def test_non_blocking_edge_does_not_block_claim(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
            # parent not done
        finally:
            p_conn.close()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child", assignee="worker")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="informed_by", blocking=False,
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                claimed = kb.claim_task(c_conn, cid, board="default")
            assert claimed is not None
            assert claimed.status == "running"
        finally:
            c_conn.close()

    def test_dangling_parent_fails_closed(self, kcd_home):
        reg = CrossBoardRegistry()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child", assignee="worker")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        # Add edge using the actual generated task id
        reg.add(
            parent_board="dangling-board", parent_id="t_dangling",
            child_board="default", child_id=cid,
            kind="blocks",
        )

        provider = CrossBoardDependencyProvider(registry=reg)
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                claimed = kb.claim_task(c_conn, cid, board="default")
            assert claimed is None
            task = kb.get_task(c_conn, cid)
            assert task.status == "todo"
            events = kb.list_events(c_conn, cid)
            rejected = [e for e in events if e.kind == "claim_rejected"]
            assert rejected
            assert rejected[-1].payload["reason"] == "external_blockers"
        finally:
            c_conn.close()


# ---------------------------------------------------------------------------
# Core seam integration: unblock after parent completion
# ---------------------------------------------------------------------------

class TestProviderUnblocksAfterCompletion:
    def test_child_promoted_after_parent_done(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent", assignee="worker")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (cid,))
        finally:
            c_conn.close()

        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="depends_on",
        )

        provider = CrossBoardDependencyProvider(registry=reg)

        # First call: parent todo → blocked
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                promoted = kb.recompute_ready(c_conn, board="default")
            assert promoted == 0
        finally:
            c_conn.close()

        # Complete parent
        p_conn = kb.connect(board="parent-board")
        try:
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()

        # Second call: parent done → child promoted
        c_conn = kb.connect(board="default")
        try:
            with scoped_dependency_provider(provider):
                promoted = kb.recompute_ready(c_conn, board="default")
            assert promoted == 1
            task = kb.get_task(c_conn, cid)
            assert task.status == "ready"
        finally:
            c_conn.close()
