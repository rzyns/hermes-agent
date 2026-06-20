from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_research_route as route


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_route_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Match dashboard plugin tests: board DBs can be rooted separately, while
    # route registers/artifacts stay under get_default_hermes_root()/artifacts.
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home / "kanban"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _make_source_task() -> str:
    kb.create_board(route.SOURCE_BOARD)
    with kb.connect(board=route.SOURCE_BOARD) as conn:
        return kb.create_task(
            conn,
            title="Assess candidate research link",
            body="Source URL: https://example.test/paper",
            created_by="test",
        )


def test_materialize_attention_route_creates_inert_durable_target(kanban_home, client):
    source_id = _make_source_task()
    url = "https://example.test/paper"

    result = route.materialize_attention_research_route(
        source_task=source_id,
        url=url,
        route_title="Read paper safely",
        route_body="Summarize only; do not clone or install.",
        source_artifact=str(kanban_home / "artifacts" / route.SOURCE_BOARD / source_id / "assessment.md"),
    )

    assert result.created is True
    assert result.source_board == route.SOURCE_BOARD
    assert result.target_board == route.TARGET_BOARD
    workspace = Path(result.target_workspace)
    assert workspace == (
        kanban_home
        / "artifacts"
        / route.TARGET_BOARD
        / "routed-attention-intake"
        / source_id
        / result.target_task
    )
    assert workspace.is_dir()

    with kb.connect(board=route.TARGET_BOARD) as conn:
        task = kb.get_task(conn, result.target_task)
        assert task is not None
        assert task.status == "blocked"
        assert task.assignee is None
        assert task.workspace_kind == "dir"
        assert task.workspace_path == str(workspace)
        row = conn.execute(
            "SELECT claim_lock, worker_pid, current_run_id FROM tasks WHERE id = ?",
            (result.target_task,),
        ).fetchone()
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None
        assert row["current_run_id"] is None
        blocked = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'blocked' ORDER BY id DESC LIMIT 1",
            (result.target_task,),
        ).fetchone()
        assert blocked is not None
        payload = json.loads(blocked["payload"])
        assert payload["source_board"] == route.SOURCE_BOARD
        assert payload["source_task"] == source_id
        assert payload["source_url"] == url

    attention_rows = _jsonl(Path(result.attention_register_jsonl))
    route_rows = [r for r in attention_rows if r.get("routed_to_task") == result.target_task]
    assert len(route_rows) == 1
    route_row = route_rows[0]
    assert route_row["routed_to_board"] == route.TARGET_BOARD
    assert route_row["source_task"] == source_id
    assert route_row["route_target_workspace"] == str(workspace)
    assert route_row["route_materialization"]["mode"] == "materialize_only"
    assert route_row["route_materialization"]["source_board"] == route.SOURCE_BOARD
    assert route_row["route_materialization"]["target_task"] == result.target_task

    target_rows = _jsonl(Path(result.target_register_jsonl))
    target_row = next(r for r in target_rows if r.get("task_id") == result.target_task)
    assert target_row["source_ref"] == f"{route.SOURCE_BOARD}/{source_id}"
    assert target_row["source_task"] == result.target_task
    assert target_row["upstream_source_task"] == source_id
    assert target_row["upstream_source_ref"] == f"{route.SOURCE_BOARD}/{source_id}"
    assert target_row["status"] == "blocked"

    response = client.patch(
        f"/api/plugins/kanban/tasks/{result.target_task}?board={route.TARGET_BOARD}",
        json={"status": "ready"},
    )
    assert response.status_code == 409
    assert "materialize_only routed target" in response.json()["detail"]


def test_materialize_attention_route_is_idempotent(kanban_home):
    source_id = _make_source_task()
    url = "https://example.test/idempotent"

    first = route.materialize_attention_research_route(source_task=source_id, url=url)
    second = route.materialize_attention_research_route(source_task=source_id, url=url)

    assert first.target_task == second.target_task
    assert first.target_workspace == second.target_workspace
    assert first.created is True
    assert second.created is False

    with kb.connect(board=route.TARGET_BOARD) as conn:
        rows = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            (f"{route.SOURCE_BOARD}:{source_id}:{route.TARGET_BOARD}:{url}",),
        ).fetchall()
    assert [r["id"] for r in rows] == [first.target_task]

    attention_rows = _jsonl(Path(first.attention_register_jsonl))
    routed_rows = [
        r for r in attention_rows
        if r.get("source_task") == source_id and r.get("routed_to_task") == first.target_task
    ]
    assert len(routed_rows) == 1
