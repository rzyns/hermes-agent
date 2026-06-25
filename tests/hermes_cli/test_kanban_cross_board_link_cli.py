"""Tests for cross-board registry mutations exposed on ``hermes kanban link``.

The canonical registry implementation already lives in the kanban-cross-deps
plugin.  These tests cover the compatibility/contract surface required by
Navigator: ``hermes kanban link/unlink`` must preserve existing board-local
positional behavior while accepting explicit cross-board flags that mutate the
canonical registry and emit machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from plugins.kanban_cross_deps.store import CrossBoardRegistry


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parse_kanban_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(["kanban", *argv])


def _run_kanban(argv: list[str]) -> int:
    return kc.kanban_command(_parse_kanban_args(argv))


def _seed_cross_board_tasks() -> tuple[str, str]:
    kb.create_board("parent-board")
    kb.create_board("child-board")
    with kb.connect_closing(board="parent-board") as conn:
        parent_id = kb.create_task(conn, title="upstream parent", initial_status="running")
        assert kb.complete_task(conn, parent_id, result="done")
    with kb.connect_closing(board="child-board") as conn:
        child_id = kb.create_task(conn, title="downstream child", assignee="worker")
    return parent_id, child_id


def test_cross_board_link_flags_create_registry_edge_json(kanban_home, capsys):
    parent_id, child_id = _seed_cross_board_tasks()

    rc = _run_kanban([
        "link",
        "--parent-board", "parent-board",
        "--parent", parent_id,
        "--child-board", "child-board",
        "--child", child_id,
        "--kind", "depends_on",
        "--required-parent-statuses", "done,archived",
        "--source", "navigator-operator",
        "--created-by", "test-operator",
        "--json",
    ])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    assert data["linked"] is True
    edge = data["edge"]
    assert edge["parent_board"] == "parent-board"
    assert edge["parent_id"] == parent_id
    assert edge["child_board"] == "child-board"
    assert edge["child_id"] == child_id
    assert edge["kind"] == "depends_on"
    assert edge["blocking"] is False
    assert edge["required_parent_statuses"] == ["done", "archived"]
    assert edge["source"] == "navigator-operator"
    assert edge["created_by"] == "test-operator"

    registry_edges = CrossBoardRegistry().list_edges(
        parent_board="parent-board",
        parent_id=parent_id,
        child_board="child-board",
        child_id=child_id,
        kind="depends_on",
    )
    assert len(registry_edges) == 1
    assert registry_edges[0].id == edge["id"]

    with kb.connect_closing(board="child-board") as conn:
        local_edges = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    assert local_edges == []


def test_cross_board_unlink_flags_remove_registry_edge_json(kanban_home, capsys):
    parent_id, child_id = _seed_cross_board_tasks()
    edge = CrossBoardRegistry().add(
        parent_board="parent-board",
        parent_id=parent_id,
        child_board="child-board",
        child_id=child_id,
        kind="depends_on",
        blocking=False,
        source="navigator-operator",
    )

    rc = _run_kanban([
        "unlink",
        "--parent-board", "parent-board",
        "--parent", parent_id,
        "--child-board", "child-board",
        "--child", child_id,
        "--kind", "depends_on",
        "--json",
    ])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data == {
        "ok": True,
        "unlinked": True,
        "edge_id": edge.id,
        "edge": {
            "parent_board": "parent-board",
            "parent_id": parent_id,
            "child_board": "child-board",
            "child_id": child_id,
            "kind": "depends_on",
        },
    }
    assert CrossBoardRegistry().get(edge.id) is None


def test_positional_link_still_creates_board_local_dependency(kanban_home, capsys):
    with kb.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="local parent", initial_status="running")
        child_id = kb.create_task(conn, title="local child", assignee="worker")

    rc = _run_kanban(["link", parent_id, child_id])

    assert rc == 0
    assert f"Linked {parent_id} -> {child_id}" in capsys.readouterr().out
    with kb.connect_closing() as conn:
        rows = conn.execute("SELECT parent_id, child_id FROM task_links").fetchall()
    assert [(row["parent_id"], row["child_id"]) for row in rows] == [(parent_id, child_id)]
    assert CrossBoardRegistry().count() == 0


def test_cross_board_link_rejects_mixed_positional_and_flag_forms(kanban_home, capsys):
    parent_id, child_id = _seed_cross_board_tasks()

    rc = _run_kanban([
        "link",
        parent_id,
        child_id,
        "--parent-board", "parent-board",
        "--parent", parent_id,
        "--child-board", "child-board",
        "--child", child_id,
        "--kind", "depends_on",
        "--json",
    ])

    assert rc == 2
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is False
    assert "choose either positional" in data["error"]
