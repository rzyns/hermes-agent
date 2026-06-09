"""Tests for kanban-cross-deps discovery: read-only candidate scanning.

Uses temp HERMES_HOME with multiple boards.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban_cross_deps.discovery import CandidateDiscovery, DependencyCandidate
from plugins.kanban_cross_deps.store import CrossBoardRegistry


def _create_board(board: str, home: Path):
    """Ensure a board DB exists and is initialized."""
    db_path = kb.kanban_db_path(board=board)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = kb.connect(db_path=db_path, board=board)
    conn.close()


@pytest.fixture
def kcd_home(tmp_path, monkeypatch):
    """Temp Hermes home with initialized kanban DB for multiple boards."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    _create_board("parent-board", home)
    _create_board("child-board", home)
    return home


@pytest.fixture
def reg():
    with __import__("tempfile").TemporaryDirectory() as tmp:
        db = Path(tmp) / "kanban" / "cross_board_dependencies.db"
        yield CrossBoardRegistry(db)


# ---------------------------------------------------------------------------
# Read-only discovery — no registry mutation
# ---------------------------------------------------------------------------

class TestDiscoveryReadOnly:
    def test_discovery_does_not_mutate_registry(self, kcd_home, reg):
        """CandidateDiscovery.discover must never add edges to the registry."""
        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child task")
            kb.update_task_body(c_conn, cid, "Depends on parent-board/t_aaaaaaaa")
        finally:
            c_conn.close()

        count_before = reg.count()
        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        count_after = reg.count()

        assert count_after == count_before
        assert len(candidates) >= 1

    def test_discovery_classifies_inferred(self, kcd_home, reg):
        """A board-qualified reference to an existing task is merely inferred."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent task")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child task")
            kb.update_task_body(c_conn, cid, f"Blocks until parent-board/{pid} is done")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        assert len(candidates) >= 1
        c = candidates[0]
        assert c.status == "inferred"
        assert c.referenced_board == "parent-board"
        assert c.referenced_id == pid
        assert c.confidence >= 0.8

    def test_discovery_classifies_already_canonical(self, kcd_home, reg):
        """If a canonical edge already exists, discovery marks it already_canonical."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent task")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child task")
            kb.update_task_body(c_conn, cid, f"Waiting on parent-board/{pid}")
        finally:
            c_conn.close()

        edge = reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="child-board", child_id=cid,
            kind="depends_on",
        )

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        canonicals = [c for c in candidates if c.status == "already_canonical"]
        assert len(canonicals) >= 1
        assert canonicals[0].canonical_edge_id == edge.id

    def test_discovery_classifies_dangling(self, kcd_home, reg):
        """A reference to a non-existent board or task is dangling."""
        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child task")
            kb.update_task_body(c_conn, cid, "Needs ghost-board/t_deadbeef")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        assert len(candidates) >= 1
        assert candidates[0].status == "dangling"
        assert candidates[0].referenced_board == "ghost-board"

    def test_discovery_skips_self_reference(self, kcd_home, reg):
        """A task referencing itself is not a candidate."""
        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="self ref")
            kb.update_task_body(c_conn, cid, f"See also child-board/{cid}")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        assert len(candidates) == 0

    def test_discovery_infers_kind_from_context(self, kcd_home, reg):
        """Kind inference should pick up contextual cues."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="decision")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(c_conn, cid, f"Decision from parent-board/{pid}")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        assert len(candidates) >= 1
        # "decision" keyword should map to depends_on_decision
        kinds = {c.inferred_kind for c in candidates}
        assert "depends_on_decision" in kinds

    def test_discovery_from_result_and_comments(self, kcd_home, reg):
        """Candidates are found in result and comments too."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            # Set body
            kb.update_task_body(c_conn, cid, "Body text")
            # Set result directly via DB since API may not expose it
            c_conn.execute(
                "UPDATE tasks SET result = ? WHERE id = ?",
                (f"Completed after parent-board/{pid}", cid),
            )
            c_conn.commit()
            # Add comment
            c_conn.execute(
                "INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
                (cid, "worker", f"Also see parent-board/{pid}", 1),
            )
            c_conn.commit()
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        sources = {c.source_location for c in candidates}
        assert "result" in sources
        assert any(s.startswith("comment:") for s in sources)

    def test_discovery_deduplicates_same_reference(self, kcd_home, reg):
        """Duplicate references in the same source should only yield one candidate."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(
                c_conn, cid,
                f"First mention parent-board/{pid} and again parent-board/{pid}"
            )
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        # Should be deduped to a single candidate for same (child,ref,source)
        assert len(candidates) == 1
        assert candidates[0].confidence == 0.9

    def test_discovery_multiple_boards(self, kcd_home, reg):
        """Discovery scoped to a specific board scans that board's tasks."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            c1 = kb.create_task(c_conn, title="child1")
            kb.update_task_body(c_conn, c1, f"needs parent-board/{pid}")
            c2 = kb.create_task(c_conn, title="child2")
            kb.update_task_body(c_conn, c2, f"needs parent-board/{pid}")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        # scan whole board
        all_cands = discovery.discover(child_board="child-board")
        assert len(all_cands) == 2
        # scan single task
        single_cands = discovery.discover(child_board="child-board", child_id=c1)
        assert len(single_cands) == 1

    def test_discovery_ambiguous_bare_task_id(self, kcd_home, reg):
        """Bare t_xxxxxxxx that exists on multiple boards is ambiguous."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        # Create same task id on child-board (impossible via normal API, so use INSERT)
        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(c_conn, cid, f"needs {pid}")
        finally:
            c_conn.close()

        # Create another board with same task id
        _create_board("other-board", kcd_home)
        o_conn = kb.connect(board="other-board")
        try:
            # Force insert same task id
            o_conn.execute(
                "INSERT OR REPLACE INTO tasks(id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at, workspace_kind, workspace_path) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pid, "dup", "", None, "todo", 0, "test", 0, None, None, "scratch", None),
            )
            o_conn.commit()
        finally:
            o_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        candidates = discovery.discover(child_board="child-board", child_id=cid)
        ambiguous = [c for c in candidates if c.status == "ambiguous"]
        assert len(ambiguous) >= 1

    def test_discovery_inferred_vs_canonical_distinct(self, kcd_home, reg):
        """Inferred candidates and canonical edges remain distinct concepts."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(c_conn, cid, f"Depends on parent-board/{pid}")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        cands = discovery.discover(child_board="child-board", child_id=cid)

        # No canonical edges yet
        assert all(c.status == "inferred" for c in cands)
        assert reg.count() == 0

        # Promotion makes canonical edge
        edge = reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="child-board", child_id=cid,
            kind="depends_on",
        )
        assert reg.count() == 1

        # Re-scan shows already_canonical
        cands2 = discovery.discover(child_board="child-board", child_id=cid)
        assert any(c.status == "already_canonical" for c in cands2)
        assert cands2[0].canonical_edge_id == edge.id

    def test_discovery_promote_candidate_explicit_only(self, kcd_home, reg):
        """Promotion must be explicit; inferred candidates never auto-promote."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(c_conn, cid, f"Depends on parent-board/{pid}")
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        cands = discovery.discover(child_board="child-board", child_id=cid)
        assert reg.count() == 0

        # Explicit promotion
        edge = reg.add(
            parent_board="parent-board", parent_id=pid,
            child_board="child-board", child_id=cid,
            kind="depends_on",
            source="promoted",
        )
        assert reg.count() == 1
        assert edge.source == "promoted"

    def test_discovery_source_locations_recorded(self, kcd_home, reg):
        """Candidates carry correct source_location values."""
        p_conn = kb.connect(board="parent-board")
        try:
            pid = kb.create_task(p_conn, title="parent")
        finally:
            p_conn.close()

        c_conn = kb.connect(board="child-board")
        try:
            cid = kb.create_task(c_conn, title="child")
            kb.update_task_body(c_conn, cid, f"body ref parent-board/{pid}")
            c_conn.execute(
                "UPDATE tasks SET result = ? WHERE id = ?",
                (f"result ref parent-board/{pid}", cid),
            )
            c_conn.commit()
        finally:
            c_conn.close()

        discovery = CandidateDiscovery(registry=reg)
        cands = discovery.discover(child_board="child-board", child_id=cid)
        body_cands = [c for c in cands if c.source_location == "body"]
        result_cands = [c for c in cands if c.source_location == "result"]
        assert len(body_cands) == 1
        assert len(result_cands) == 1
