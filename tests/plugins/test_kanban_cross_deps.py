"""Tests for the kanban-cross-deps plugin registry/storage slice.

Uses temp HERMES_HOME so no live board state is touched.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from plugins.kanban_cross_deps.models import CrossBoardEdge, VALID_EDGE_KINDS, _utc_now
from plugins.kanban_cross_deps.store import CrossBoardRegistry


@pytest.fixture
def reg(monkeypatch):
    """Registry backed by a temp DB."""
    # Prevent leaked kanban overrides from affecting board/registry resolution
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "kanban" / "cross_board_dependencies.db"
        registry = CrossBoardRegistry(db)
        yield registry


@pytest.fixture
def reg2(monkeypatch):
    """Second registry pointing at the same temp DB for multi-instance tests."""
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "kanban" / "cross_board_dependencies.db"
        r1 = CrossBoardRegistry(db)
        r2 = CrossBoardRegistry(db)
        yield (r1, r2)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestCrossBoardEdgeModel:
    def test_valid_edge(self):
        e = CrossBoardEdge(
            id="x123",
            parent_board="a",
            parent_id="p1",
            child_board="b",
            child_id="c1",
            kind="blocks",
        )
        assert e.blocking is True
        assert e.required_parent_statuses == ["done", "archived"]
        assert e.source == "canonical"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            CrossBoardEdge(
                id="x123",
                parent_board="a",
                parent_id="p1",
                child_board="b",
                child_id="c1",
                kind="invalid_kind",
            )

    def test_missing_board_raises(self):
        with pytest.raises(ValueError, match="parent_board is required"):
            CrossBoardEdge(
                id="x123",
                parent_board="",
                parent_id="p1",
                child_board="b",
                child_id="c1",
                kind="blocks",
            )

    def test_to_dict_roundtrip(self):
        e = CrossBoardEdge(
            id="x123",
            parent_board="a",
            parent_id="p1",
            child_board="b",
            child_id="c1",
            kind="depends_on_decision",
            blocking=False,
            required_parent_statuses=["done"],
            source="test",
            created_by="alice",
            metadata={"priority": "high"},
        )
        d = e.to_dict()
        assert d["kind"] == "depends_on_decision"
        assert d["blocking"] is False
        assert d["required_parent_statuses"] == ["done"]
        assert d["metadata"]["priority"] == "high"

    def test_from_row_parses_json_fields(self):
        row = {
            "id": "x1",
            "parent_board": "pb",
            "parent_id": "pid",
            "child_board": "cb",
            "child_id": "cid",
            "kind": "informed_by",
            "blocking": 0,
            "required_parent_statuses": '["done"]',
            "source": "canonical",
            "created_by": None,
            "created_at": 1717900000,
            "updated_at": 1717900001,
            "metadata": '{"note": "hello"}',
        }
        e = CrossBoardEdge.from_row(row)
        assert e.blocking is False
        assert e.required_parent_statuses == ["done"]
        assert e.metadata == {"note": "hello"}
        assert e.created_at is not None
        assert e.updated_at is not None


# ---------------------------------------------------------------------------
# Registry idempotent init / schema
# ---------------------------------------------------------------------------

class TestRegistryInit:
    def test_init_creates_db_and_tables(self, reg):
        reg._ensure_init()
        assert reg.path.exists()
        with reg._conn() as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row["name"] for row in cur.fetchall()}
        assert "cross_board_edges" in tables
        assert "_registry_meta" in tables

    def test_schema_version_set(self, reg):
        reg._ensure_init()
        assert reg.schema_version() == 1

    def test_init_is_idempotent(self, reg):
        reg._ensure_init()
        reg._ensure_init()
        assert reg.schema_version() == 1


# ---------------------------------------------------------------------------
# Add / get / remove
# ---------------------------------------------------------------------------

class TestRegistryAddGetRemove:
    def test_add_and_get(self, reg):
        e = reg.add(
            parent_board="research",
            parent_id="t_parent",
            child_board="engineering",
            child_id="t_child",
            kind="blocks",
        )
        assert e.id.startswith("x")
        fetched = reg.get(e.id)
        assert fetched is not None
        assert fetched.parent_board == "research"
        assert fetched.child_board == "engineering"
        assert fetched.blocking is True

    def test_add_duplicate_raises(self, reg):
        reg.add(
            parent_board="a",
            parent_id="p1",
            child_board="b",
            child_id="c1",
            kind="blocks",
        )
        with pytest.raises(ValueError, match="Edge already exists"):
            reg.add(
                parent_board="a",
                parent_id="p1",
                child_board="b",
                child_id="c1",
                kind="blocks",
            )

    def test_add_different_kind_same_pair_allowed(self, reg):
        e1 = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
        )
        e2 = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="informed_by",
        )
        assert e1.id != e2.id

    def test_remove_by_id(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
        )
        assert reg.remove(e.id) is True
        assert reg.get(e.id) is None
        assert reg.remove(e.id) is False

    def test_remove_by_composite(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
        )
        assert reg.remove_by_composite(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
        ) is True
        assert reg.get(e.id) is None

    def test_all_kinds_accepted(self, reg):
        for i, kind in enumerate(VALID_EDGE_KINDS):
            e = reg.add(
                parent_board="a", parent_id=f"p{i}",
                child_board="b", child_id=f"c{i}",
                kind=kind,
            )
            assert e.kind == kind

    def test_add_with_custom_fields(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="depends_on",
            blocking=False,
            required_parent_statuses=["done"],
            source="manual",
            created_by="alice",
            metadata={"note": "hello"},
        )
        fetched = reg.get(e.id)
        assert fetched.blocking is False
        assert fetched.required_parent_statuses == ["done"]
        assert fetched.source == "manual"
        assert fetched.created_by == "alice"
        assert fetched.metadata == {"note": "hello"}


# ---------------------------------------------------------------------------
# List / filter / count
# ---------------------------------------------------------------------------

class TestRegistryListFilter:
    def test_list_all(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks")
        assert len(reg.list_edges()) == 2

    def test_filter_by_child(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks")
        reg.add(parent_board="a", parent_id="p3", child_board="c", child_id="c3", kind="blocks")
        results = reg.list_edges(child_board="b")
        assert len(results) == 2
        assert all(e.child_board == "b" for e in results)

    def test_filter_by_blocking(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks", blocking=True)
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="informed_by", blocking=False)
        assert len(reg.list_edges(blocking=True)) == 1
        assert len(reg.list_edges(blocking=False)) == 1

    def test_filter_by_source(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks", source="canonical")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks", source="inferred")
        assert len(reg.list_edges(source="canonical")) == 1
        assert len(reg.list_edges(source="inferred")) == 1

    def test_count(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks")
        assert reg.count() == 2
        assert reg.count(child_board="b") == 2
        assert reg.count(child_board="z") == 0

    def test_list_pagination(self, reg):
        for i in range(10):
            reg.add(parent_board="a", parent_id=f"p{i}", child_board="b", child_id=f"c{i}", kind="blocks")
        page = reg.list_edges(limit=3, offset=2)
        assert len(page) == 3


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

class TestRegistryUpdate:
    def test_update_metadata_merge(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
            metadata={"a": 1},
        )
        updated = reg.update_metadata(e.id, {"b": 2}, merge=True)
        assert updated is not None
        assert updated.metadata == {"a": 1, "b": 2}

    def test_update_metadata_replace(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
            metadata={"a": 1},
        )
        updated = reg.update_metadata(e.id, {"b": 2}, merge=False)
        assert updated.metadata == {"b": 2}

    def test_update_blocking(self, reg):
        e = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks", blocking=True,
        )
        updated = reg.update_blocking(e.id, False)
        assert updated is not None
        assert updated.blocking is False
        fetched = reg.get(e.id)
        assert fetched.blocking is False

    def test_update_missing_edge_returns_none(self, reg):
        assert reg.update_metadata("x_missing", {"a": 1}) is None
        assert reg.update_blocking("x_missing", False) is None


# ---------------------------------------------------------------------------
# Multi-instance / thread safety smoke
# ---------------------------------------------------------------------------

class TestRegistryConcurrency:
    def test_second_instance_sees_first_instance_writes(self, reg2):
        r1, r2 = reg2
        e = r1.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        fetched = r2.get(e.id)
        assert fetched is not None
        assert fetched.parent_board == "a"

    def test_reset_clears_all(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.reset()
        assert reg.count() == 0
        assert reg.schema_version() == 0


# ---------------------------------------------------------------------------
# Cycle guard
# ---------------------------------------------------------------------------

class TestRegistryCycleGuard:
    def test_self_loop_rejected(self, reg):
        with pytest.raises(ValueError, match="blocking cycle"):
            reg.add(
                parent_board="a", parent_id="x",
                child_board="a", child_id="x",
                kind="blocks", blocking=True,
            )

    def test_blocking_cycle_rejected(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        with pytest.raises(ValueError, match="blocking cycle"):
            reg.add(
                parent_board="b", parent_id="c1",
                child_board="a", child_id="p1",
                kind="blocks", blocking=True,
            )

    def test_non_blocking_cycle_allowed(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="related", blocking=False)
        reg.add(
            parent_board="b", parent_id="c1",
            child_board="a", child_id="p1",
            kind="related", blocking=False,
        )
        assert reg.count() == 2

    def test_reject_cycle_false_allows_cycle(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(
            parent_board="b", parent_id="c1",
            child_board="a", child_id="p1",
            kind="blocks", reject_cycle=False,
        )
        assert reg.count() == 2

    def test_cycle_via_transitive_blocking_rejected(self, reg):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(parent_board="b", parent_id="c1", child_board="c", child_id="d1", kind="blocks")
        with pytest.raises(ValueError, match="blocking cycle"):
            reg.add(
                parent_board="c", parent_id="d1",
                child_board="a", child_id="p1",
                kind="blocks", blocking=True,
            )

    def test_local_plus_cross_cycle_rejected(self, reg, monkeypatch):
        """A new cross-board edge closing a cycle through existing local links must be rejected.

        Graph: local A -> B (task_links), cross B -> C, then attempted cross C -> A.
        The final edge must raise before writing.
        """
        from hermes_cli import kanban_db as _kb

        # Use a real board DB so local task_links can be created.
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
        # Re-init with fresh home so connect() resolves to our temp root.
        _kb.init_db(board="local_cycle_board")
        conn = _kb.connect(board="local_cycle_board")
        try:
            a = _kb.create_task(conn, title="A", assignee="w")
            b = _kb.create_task(conn, title="B", assignee="w")
            c = _kb.create_task(conn, title="C", assignee="w")
            _kb.link_tasks(conn, a, b)  # local A -> B
        finally:
            conn.close()

        # cross-board B -> C
        reg.add(
            parent_board="local_cycle_board", parent_id=b,
            child_board="x", child_id=c,
            kind="blocks", blocking=True,
        )

        # cross-board C -> A would close the cycle; must reject
        with pytest.raises(ValueError, match="blocking cycle"):
            reg.add(
                parent_board="x", parent_id=c,
                child_board="local_cycle_board", child_id=a,
                kind="blocks", blocking=True,
            )

        # Only the first cross-board edge should have been written.
        assert reg.count() == 1

    def test_update_blocking_promotion_rejects_cycle(self, reg, monkeypatch):
        """Turning an existing non-blocking edge into blocking must also guard cycles.

        Graph: local A -> B, cross B -> C, cross C -> A (non-blocking).
        Promoting C -> A to blocking must raise because it would create a cycle.
        """
        from hermes_cli import kanban_db as _kb

        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
        _kb.init_db(board="promote_cycle_board")
        conn = _kb.connect(board="promote_cycle_board")
        try:
            a = _kb.create_task(conn, title="A", assignee="w")
            b = _kb.create_task(conn, title="B", assignee="w")
            c = _kb.create_task(conn, title="C", assignee="w")
            _kb.link_tasks(conn, a, b)
        finally:
            conn.close()

        reg.add(
            parent_board="promote_cycle_board", parent_id=b,
            child_board="x", child_id=c,
            kind="blocks", blocking=True,
        )
        edge = reg.add(
            parent_board="x", parent_id=c,
            child_board="promote_cycle_board", child_id=a,
            kind="blocks", blocking=False,
        )

        with pytest.raises(ValueError, match="Promoting edge .* to blocking would create a cycle"):
            reg.update_blocking(edge.id, blocking=True)

    def test_read_only_no_db_creation(self, reg):
        """Read-only ops on a registry that was never written must not create the DB."""
        assert not reg.path.exists()
        assert reg.get("missing") is None
        assert reg.list_edges() == []
        assert reg.count() == 0
        assert reg.schema_version() == 0
        assert not reg.path.exists()


# ---------------------------------------------------------------------------
# Profile-mode regression: shared kanban root anchoring
# ---------------------------------------------------------------------------

class TestRegistryRootAnchoring:
    def test_registry_defaults_to_kanban_home(self, monkeypatch, tmp_path):
        """When no explicit db_path is given, CrossBoardRegistry anchors to
        kanban_home(), not get_hermes_home(), so it stays visible across
        profiles that share boards.
        """
        shared = tmp_path / "shared_kanban"
        shared.mkdir()
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(shared))
        monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
        monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
        # Set an isolated profile home
        profile_home = tmp_path / "profiles" / "some_profile"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setattr(Path, "home", lambda: profile_home)

        reg = CrossBoardRegistry()
        expected = shared / "kanban" / "cross_board_dependencies.db"
        assert reg.path == expected
        edge = reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        assert reg.path.exists()

        # Simulate another profile reading the same shared root
        profile_home2 = tmp_path / "profiles" / "other_profile"
        profile_home2.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home2))
        monkeypatch.setattr(Path, "home", lambda: profile_home2)
        reg2 = CrossBoardRegistry()
        assert reg2.path == expected
        fetched = reg2.get(edge.id)
        assert fetched is not None
        assert fetched.parent_board == "a"
