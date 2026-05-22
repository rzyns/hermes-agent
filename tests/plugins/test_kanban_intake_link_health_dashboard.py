"""Tests for the dashboard GET /intake-links/health endpoint (plugin_api.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_intake_link as kil


def _load_plugin_router():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py and return its router."""
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_test_health", plugin_file,
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
# GET /intake-links/health
# ---------------------------------------------------------------------------


def test_health_empty_board(client):
    """Scan mode on empty board → zero counts."""
    r = client.get("/api/plugins/kanban/intake-links/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["scanned_task_count"] == 0
    assert data["counts"]["provisionally_registered"] == 0


def test_health_task_not_found(client):
    """Single-task mode with unknown id → 404."""
    r = client.get("/api/plugins/kanban/intake-links/health?task_id=t_nonexistent")
    assert r.status_code == 404


def test_health_single_task(client, kanban_home):
    """Single-task mode returns register-health dict."""
    # Create a task via the creation path so body includes contract.
    r = client.post("/api/plugins/kanban/intake-links", json={"url": "https://example.com/h"})
    assert r.status_code == 200
    task = r.json()["task"]
    tid = task["id"]

    rid = client.get(f"/api/plugins/kanban/intake-links/health?task_id={tid}")
    assert rid.status_code == 200
    data = rid.json()
    assert data["task_id"] == tid
    assert data["verdict"] == "provisional_only"
    assert data["has_provisional_entry"] is True
    assert data["body_contract_ok"] is True
    assert data["url"] == "https://example.com/h"


def test_health_scan_with_task(client, kanban_home):
    """Board scan covers the created task."""
    r = client.post("/api/plugins/kanban/intake-links", json={"url": "https://example.com/s"})
    assert r.status_code == 200

    rid = client.get("/api/plugins/kanban/intake-links/health")
    assert rid.status_code == 200
    data = rid.json()
    assert data["scanned_task_count"] == 1
    assert data["counts"]["provisional_only"] == 1
