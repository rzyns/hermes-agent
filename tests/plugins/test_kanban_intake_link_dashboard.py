"""Tests for the dashboard POST /intake-links endpoint (plugin_api.py).

Uses the same bare-FastAPI harness as
``tests/plugins/test_kanban_dashboard_plugin.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("attention-intake")
    return home


@pytest.fixture
def client(kanban_home):
    router = _load_plugin_router()
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/kanban")
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /intake-links
# ---------------------------------------------------------------------------


def test_create_intake_link_basic(client):
    r = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/article",
        "context": "Great read",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert "task" in data
    task = data["task"]
    assert task["title"].startswith("Link drop:")
    assert "https://example.com/article" in task["body"]
    assert task["status"] == "triage"
    assert task["assignee"] == "link-analyst"
    assert task["workspace_path"] is not None
    # Ensure workspace dir was created.
    assert Path(task["workspace_path"]).exists()


def test_create_intake_link_dedup(client):
    r1 = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/dedup",
    })
    tid1 = r1.json()["task"]["id"]
    r2 = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/dedup",
    })
    tid2 = r2.json()["task"]["id"]
    assert tid1 == tid2


def test_create_intake_link_custom_assignee_priority(client):
    r = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/custom",
        "assignee": "researcher-a",
        "priority": 7,
    })
    task = r.json()["task"]
    assert task["assignee"] == "researcher-a"
    assert task["priority"] == 7


def test_create_intake_link_empty_url_bad_request(client):
    r = client.post("/api/plugins/kanban/intake-links", json={"url": ""})
    assert r.status_code == 400


def test_create_intake_link_board_override(client):
    """Query-string board=default must be ignored; card always lands on attention-intake."""
    r = client.post("/api/plugins/kanban/intake-links?board=default", json={
        "url": "https://example.com/other",
    })
    assert r.status_code == 200, r.text
    task = r.json()["task"]
    # The card MUST be on attention-intake, not "default".
    # We verify by loading the task directly from the attention-intake connection
    from hermes_cli import kanban_db as kb
    with kb.connect(board="attention-intake") as conn:
        row = kb.get_task(conn, task["id"])
        assert row is not None


def test_dashboard_targets_attention_intake_not_selected_board(client):
    """Regression: dashboard Drop Link must target attention-intake even when
    the active board in the UI is something else."""
    # Create another board and a task there so we can verify separation.
    from hermes_cli import kanban_db as kb
    with kb.connect(board="attention-intake") as conn:
        kb.create_board("other-board")
    r = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/dashboard-target",
    })
    assert r.status_code == 200
    task = r.json()["task"]
    with kb.connect(board="attention-intake") as conn:
        row = kb.get_task(conn, task["id"])
        assert row is not None
        assert "attention-intake" in row.body
    # It must NOT appear on "other-board"
    with kb.connect(board="other-board") as conn:
        row = kb.get_task(conn, task["id"])
        assert row is None
