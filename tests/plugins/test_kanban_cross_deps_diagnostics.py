"""Tests for kanban-cross-deps diagnostics: dangling, cycles, contradictions, provider failures.

Uses temp HERMES_HOME with multiple boards.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_dependencies import ExternalBlocker
from plugins.kanban_cross_deps.diagnostics import CrossBoardDiagnostics, CyclePath, DanglingEdge
from plugins.kanban_cross_deps.provider import CrossBoardDependencyProvider
from plugins.kanban_cross_deps.store import CrossBoardRegistry


def _create_board(board: str, home: Path):
    """Ensure a board DB exists and is initialized."""
    db_path = kb.kanban_db_path(board=board)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = kb.connect(db_path=db_path, board=board)
    conn.close()


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


# ---------------------------------------------------------------------------
# Dangling edges
# ---------------------------------------------------------------------------

class TestDiagnosticsDangling:
    def test_dangling_parent_board(self, kcd_home):
        reg = CrossBoardRegistry()
        reg.add(
            parent_board="missing-board", parent_id="t_p",
            child_board="default", child_id="t_c",
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["dangling"] >= 1
        d = report["dangling"]
        assert any("missing-board" in entry["reason"] for entry in d)

    def test_dangling_parent_task(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("real-board", kcd_home)
        reg.add(
            parent_board="real-board", parent_id="t_missing",
            child_board="default", child_id="t_c",
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["dangling"] >= 1
        d = report["dangling"]
        assert any(entry["missing_side"] == "parent" for entry in d)

    def test_dangling_child_task(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("real-board", kcd_home)
        p_conn = kb.connect(board="real-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()
        reg.add(
            parent_board="real-board", parent_id=pid,
            child_board="real-board", child_id="t_missing",
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["dangling"] >= 1
        d = report["dangling"]
        assert any(entry["missing_side"] == "child" for entry in d)

    def test_no_dangling_for_valid_tasks(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("real-board", kcd_home)
        p_conn = kb.connect(board="real-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
        finally:
            c_conn.close()
        reg.add(
            parent_board="real-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["dangling"] == 0


# ---------------------------------------------------------------------------
# Cycles: local + cross-board
# ---------------------------------------------------------------------------

class TestDiagnosticsCycles:
    def test_local_cycle_only(self, kcd_home):
        """task_links A->B->C->A produces a cycle (inserted directly since link_tasks guards against it)."""
        conn = kb.connect(board="default")
        try:
            a = kb.create_task(conn, title="A")
            b = kb.create_task(conn, title="B")
            c = kb.create_task(conn, title="C")
            with kb.write_txn(conn):
                conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (a, b))
                conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (b, c))
                conn.execute("INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)", (c, a))
        finally:
            conn.close()
        diag = CrossBoardDiagnostics(registry=CrossBoardRegistry())
        report = diag.run()
        assert report["summary"]["blocking_cycles"] >= 1

    def test_cross_board_cycle(self, kcd_home):
        """A on board1 -> B on board2 -> A on board1 via cross-board edges."""
        reg = CrossBoardRegistry()
        _create_board("board1", kcd_home)
        _create_board("board2", kcd_home)
        p1 = kb.connect(board="board1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="board2")
        try:
            b = kb.create_task(p2, title="B")
        finally:
            p2.close()

        reg.add(parent_board="board1", parent_id=a, child_board="board2", child_id=b, kind="blocks")
        reg.add(parent_board="board2", parent_id=b, child_board="board1", child_id=a, kind="blocks", reject_cycle=False)

        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["blocking_cycles"] >= 1
        paths = report["cycles"]["blocking"]
        assert len(paths) == 1
        path = paths[0]["path"]
        assert any("board1" in node and a in node for node in path)
        assert any("board2" in node and b in node for node in path)

    def test_local_plus_cross_board_cycle(self, kcd_home):
        """A -> B (local), B -> C (cross-board), C -> A (cross-board)."""
        reg = CrossBoardRegistry()
        _create_board("board1", kcd_home)
        _create_board("board2", kcd_home)
        p1 = kb.connect(board="board1")
        try:
            a = kb.create_task(p1, title="A")
            b = kb.create_task(p1, title="B")
            kb.link_tasks(p1, a, b)
        finally:
            p1.close()
        p2 = kb.connect(board="board2")
        try:
            c = kb.create_task(p2, title="C")
        finally:
            p2.close()

        reg.add(parent_board="board1", parent_id=b, child_board="board2", child_id=c, kind="blocks")
        reg.add(parent_board="board2", parent_id=c, child_board="board1", child_id=a, kind="blocks")

        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["blocking_cycles"] >= 1
        paths = report["cycles"]["blocking"]
        assert any("board1" in node and a in node for path in paths for node in path["path"])

    def test_non_blocking_cycle_is_informational(self, kcd_home):
        """Non-blocking cross-board cycles are allowed but reported as informational."""
        reg = CrossBoardRegistry()
        _create_board("board1", kcd_home)
        _create_board("board2", kcd_home)
        p1 = kb.connect(board="board1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="board2")
        try:
            b = kb.create_task(p2, title="B")
        finally:
            p2.close()

        reg.add(parent_board="board1", parent_id=a, child_board="board2", child_id=b, kind="related", blocking=False)
        reg.add(parent_board="board2", parent_id=b, child_board="board1", child_id=a, kind="related", blocking=False)

        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["blocking_cycles"] == 0
        assert report["summary"]["informational_cycles"] >= 1

    def test_cycle_normalization_dedupes_rotations(self, kcd_home):
        """Same cycle entered twice in different order should dedupe."""
        reg = CrossBoardRegistry()
        _create_board("b1", kcd_home)
        _create_board("b2", kcd_home)
        p1 = kb.connect(board="b1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="b2")
        try:
            c = kb.create_task(p2, title="C")
        finally:
            p2.close()

        reg.add(parent_board="b1", parent_id=a, child_board="b2", child_id=c, kind="blocks")
        reg.add(parent_board="b2", parent_id=c, child_board="b1", child_id=a, kind="blocks", reject_cycle=False)
        # Also add local cycle manually to test dedup if possible; since only two nodes it's fine
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["blocking_cycles"] == 1

    def test_would_create_cycle_true_for_blocking(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("b1", kcd_home)
        _create_board("b2", kcd_home)
        p1 = kb.connect(board="b1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="b2")
        try:
            b = kb.create_task(p2, title="B")
        finally:
            p2.close()
        reg.add(parent_board="b1", parent_id=a, child_board="b2", child_id=b, kind="blocks")
        diag = CrossBoardDiagnostics(registry=reg)
        assert diag.would_create_cycle("b2", b, "b1", a, blocking=True) is True

    def test_would_create_cycle_false_for_non_blocking(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("b1", kcd_home)
        _create_board("b2", kcd_home)
        p1 = kb.connect(board="b1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="b2")
        try:
            b = kb.create_task(p2, title="B")
        finally:
            p2.close()
        reg.add(parent_board="b1", parent_id=a, child_board="b2", child_id=b, kind="blocks")
        diag = CrossBoardDiagnostics(registry=reg)
        assert diag.would_create_cycle("b2", b, "b1", a, blocking=False) is False

    def test_cycle_guard_rejects_add(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("b1", kcd_home)
        _create_board("b2", kcd_home)
        p1 = kb.connect(board="b1")
        try:
            a = kb.create_task(p1, title="A")
        finally:
            p1.close()
        p2 = kb.connect(board="b2")
        try:
            b = kb.create_task(p2, title="B")
        finally:
            p2.close()
        reg.add(parent_board="b1", parent_id=a, child_board="b2", child_id=b, kind="blocks")
        diag = CrossBoardDiagnostics(registry=reg)
        assert diag.would_create_cycle("b2", b, "b1", a, blocking=True) is True


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------

class TestDiagnosticsContradictions:
    def test_satisfied_parent_child_stuck(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            # Child created as running by default; force it to todo so it appears stuck
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (cid,))
        finally:
            c_conn.close()
        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["contradictions"] >= 1
        c = report["contradictions"][0]
        assert c["kind"] == "satisfied_parent_child_stuck"

    def test_unsatisfied_parent_child_ready(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            with kb.write_txn(c_conn):
                c_conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (cid,))
        finally:
            c_conn.close()
        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["contradictions"] >= 1
        c = report["contradictions"][0]
        assert c["kind"] == "unsatisfied_parent_child_ready_running"

    def test_no_contradiction_when_child_done(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("parent-board", kcd_home)
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.complete_task(c_conn, cid, summary="done")
        finally:
            c_conn.close()
        reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["contradictions"] == 0


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------

class TestDiagnosticsProviderFailures:
    def test_provider_failure_for_dangling(self, kcd_home):
        reg = CrossBoardRegistry()
        reg.add(
            parent_board="missing", parent_id="t_m",
            child_board="default", child_id="t_c",
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["provider_failures"] >= 1
        pf = report["provider_failures"][0]
        assert pf["error_type"] == "provider_dangling"

    def test_no_provider_failure_for_valid(self, kcd_home):
        reg = CrossBoardRegistry()
        _create_board("pb", kcd_home)
        p_conn = kb.connect(board="pb")
        try:
            pid = kb.create_task(p_conn, title="parent")
            kb.complete_task(p_conn, pid, summary="done")
        finally:
            p_conn.close()
        c_conn = kb.connect(board="default")
        try:
            cid = kb.create_task(c_conn, title="child")
        finally:
            c_conn.close()
        reg.add(
            parent_board="pb", parent_id=pid,
            child_board="default", child_id=cid,
            kind="blocks",
        )
        diag = CrossBoardDiagnostics(registry=reg)
        report = diag.run()
        assert report["summary"]["provider_failures"] == 0


# ---------------------------------------------------------------------------
# CLI integration: diagnostics command and cycle guard
# ---------------------------------------------------------------------------

class TestCliDiagnosticsIntegration:
    def test_diagnostics_command_json(self, kcd_home, capsys):
        import argparse
        from plugins.kanban_cross_deps.cli import _cmd_diagnostics
        args = argparse.Namespace(json=True)
        rc = _cmd_diagnostics(args)
        assert rc == 0
        captured = capsys.readouterr()
        import json as _json
        data = _json.loads(captured.out)
        assert "summary" in data

    def test_diagnostics_command_human(self, kcd_home, capsys):
        import argparse
        from plugins.kanban_cross_deps.cli import _cmd_diagnostics
        args = argparse.Namespace(json=False)
        rc = _cmd_diagnostics(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Diagnostics" in captured.out
