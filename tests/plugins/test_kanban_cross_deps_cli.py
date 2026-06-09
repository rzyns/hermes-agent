"""Tests for the kanban-cross-deps plugin CLI surface.

Uses temp HERMES_HOME so no live board state is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.kanban_cross_deps.cli import (
    _cmd_add,
    _cmd_list,
    _cmd_remove,
    _cmd_status,
    kanban_cross_deps_command,
    register_cli,
)
from plugins.kanban_cross_deps.store import CrossBoardRegistry


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a Namespace with all CLI defaults."""
    defaults = {
        "kcd_command": None,
        "parent_board": None,
        "parent_id": None,
        "child_board": None,
        "child_id": None,
        "kind": None,
        "blocking": True,
        "required_statuses": None,
        "source": "canonical",
        "created_by": None,
        "metadata": None,
        "id": None,
        "json": False,
        "limit": 500,
        "offset": 0,
        "count": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture
def reg():
    """Registry backed by an explicit temp DB path."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "kanban" / "cross_board_dependencies.db"
        yield CrossBoardRegistry(db)


# ---------------------------------------------------------------------------
# register_cli smoke
# ---------------------------------------------------------------------------

class TestRegisterCli:
    def test_builds_subparsers(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        kcd_parser = sub.add_parser("kanban-cross-deps")
        register_cli(kcd_parser)
        args = parser.parse_args([
            "kanban-cross-deps", "add",
            "--parent-board", "a", "--parent-id", "p",
            "--child-board", "b", "--child-id", "c",
            "--kind", "blocks",
        ])
        assert args.kcd_command == "add"
        assert args.parent_board == "a"
        assert args.kind == "blocks"


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

class TestCliAdd:
    def test_add_basic(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="research",
            parent_id="t_p",
            child_board="eng",
            child_id="t_c",
            kind="blocks",
        )
        rc = _cmd_add(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        assert "Added edge" in captured.out
        assert "research/t_p --[blocks] (blocking)--> eng/t_c" in captured.out

    def test_add_json(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="depends_on",
            blocking=False,
            json=True,
        )
        rc = _cmd_add(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["edge"]["kind"] == "depends_on"
        assert data["edge"]["blocking"] is False

    def test_add_invalid_kind_rejected(self, reg, capsys):
        # argparse should reject invalid choices before the handler ever runs.
        with pytest.raises(SystemExit):
            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers()
            kcd_parser = sub.add_parser("kanban-cross-deps")
            register_cli(kcd_parser)
            parser.parse_args([
                "kanban-cross-deps", "add",
                "--parent-board", "a", "--parent-id", "p",
                "--child-board", "b", "--child-id", "c",
                "--kind", "invalid_kind",
            ])

    def test_add_duplicate_rejected(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
        )
        assert _cmd_add(args, reg) == 0
        rc = _cmd_add(args, reg)
        assert rc == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.out or "already exists" in captured.err

    def test_add_with_metadata_and_statuses(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
            required_statuses='["done"]',
            metadata='{"note": "hello"}',
            json=True,
        )
        rc = _cmd_add(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["edge"]["required_parent_statuses"] == ["done"]
        assert data["edge"]["metadata"]["note"] == "hello"

    def test_add_invalid_json_metadata(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
            metadata="not json",
            json=True,
        )
        rc = _cmd_add(args, reg)
        assert rc == 2
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "metadata" in data["error"].lower()

    def test_add_invalid_json_statuses(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
            required_statuses="not-a-list",
            json=True,
        )
        rc = _cmd_add(args, reg)
        assert rc == 2
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "required" in data["error"].lower()

    def test_add_provenance(self, reg, capsys):
        args = _make_args(
            kcd_command="add",
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
            source="manual",
            created_by="alice",
            json=True,
        )
        rc = _cmd_add(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["edge"]["source"] == "manual"
        assert data["edge"]["created_by"] == "alice"


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

class TestCliRemove:
    def test_remove_by_id(self, reg, capsys):
        e = reg.add(parent_board="a", parent_id="p", child_board="b", child_id="c", kind="blocks")
        args = _make_args(kcd_command="remove", id=e.id, json=True)
        rc = _cmd_remove(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert reg.get(e.id) is None

    def test_remove_by_composite(self, reg, capsys):
        e = reg.add(parent_board="a", parent_id="p", child_board="b", child_id="c", kind="blocks")
        args = _make_args(
            kcd_command="remove",
            id=None,
            parent_board="a",
            parent_id="p",
            child_board="b",
            child_id="c",
            kind="blocks",
            json=True,
        )
        rc = _cmd_remove(args, reg)
        assert rc == 0
        assert reg.get(e.id) is None

    def test_remove_missing_id_and_composite(self, reg, capsys):
        args = _make_args(kcd_command="remove", id=None, parent_board="a", json=True)
        rc = _cmd_remove(args, reg)
        assert rc == 2
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "requires --id OR all" in data["error"]

    def test_remove_not_found(self, reg, capsys):
        args = _make_args(kcd_command="remove", id="x_nonexistent", json=True)
        rc = _cmd_remove(args, reg)
        assert rc == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestCliList:
    def test_list_empty(self, reg, capsys):
        args = _make_args(kcd_command="list", json=True)
        rc = _cmd_list(args, reg)
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["edges"] == []
        assert data["count"] == 0

    def test_list_filter_by_child(self, reg, capsys):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks")
        reg.add(parent_board="a", parent_id="p3", child_board="c", child_id="c3", kind="blocks")
        args = _make_args(kcd_command="list", child_board="b", json=True)
        rc = _cmd_list(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 2
        for e in data["edges"]:
            assert e["child_board"] == "b"

    def test_list_count_only(self, reg, capsys):
        reg.add(parent_board="a", parent_id="p", child_board="b", child_id="c", kind="blocks")
        args = _make_args(kcd_command="list", count=True, json=True)
        rc = _cmd_list(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1

    def test_list_filter_by_source(self, reg, capsys):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c1", kind="blocks", source="canonical")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c2", kind="blocks", source="inferred")
        args = _make_args(kcd_command="list", source="canonical", json=True)
        rc = _cmd_list(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 1
        assert data["edges"][0]["source"] == "canonical"

    def test_list_pagination(self, reg, capsys):
        for i in range(5):
            reg.add(parent_board="a", parent_id=f"p{i}", child_board="b", child_id=f"c{i}", kind="blocks")
        args = _make_args(kcd_command="list", limit=2, offset=1, json=True)
        rc = _cmd_list(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["count"] == 2


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestCliStatus:
    def test_status_no_edges(self, reg, capsys):
        args = _make_args(kcd_command="status", child_board="b", child_id="c", json=True)
        rc = _cmd_status(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_edges"] == 0
        assert data["blocking_edges"] == 0
        assert data["canonical_edges"] == 0

    def test_status_with_mixed_edges(self, reg, capsys):
        reg.add(parent_board="a", parent_id="p1", child_board="b", child_id="c", kind="blocks", blocking=True, source="canonical")
        reg.add(parent_board="a", parent_id="p2", child_board="b", child_id="c", kind="informed_by", blocking=False, source="inferred")
        args = _make_args(kcd_command="status", child_board="b", child_id="c", json=True)
        rc = _cmd_status(args, reg)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_edges"] == 2
        assert data["blocking_edges"] == 1
        assert data["non_blocking_edges"] == 1
        assert data["canonical_edges"] == 1
        assert data["inferred_edges"] == 1
        assert len(data["edges"]) == 2


# ---------------------------------------------------------------------------
# top-level dispatch
# ---------------------------------------------------------------------------

class TestTopLevelDispatch:
    def test_no_subcommand(self, capsys):
        args = _make_args()
        rc = kanban_cross_deps_command(args)
        assert rc == 2
        assert "usage:" in capsys.readouterr().out

    def test_unknown_subcommand(self, capsys):
        args = _make_args(kcd_command="xyz")
        rc = kanban_cross_deps_command(args)
        assert rc == 2
        assert "Unknown" in capsys.readouterr().err
