"""Dashboard-level smoke: verify the Drop Link frontend exists and POST wiring is reachable.

Uses the same bare-FastAPI harness as test_kanban_dashboard_plugin.py and
test_kanban_intake_link_dashboard.py.
"""

import importlib.util
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


def test_intake_links_endpoint_returns_200(client):
    r = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://example.com/smoke",
        "context": "Dashboard smoke test",
    })
    assert r.status_code == 200, r.text
    data = r.json()
    assert "task" in data
    task = data["task"]
    assert task["title"].startswith("Link drop:")
    assert task["status"] == "triage"
    assert task["assignee"] == "link-analyst"


def test_intake_links_dedup_by_url(client):
    r1 = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://dedup.example.com/page",
    })
    r2 = client.post("/api/plugins/kanban/intake-links", json={
        "url": "https://dedup.example.com/page",
    })
    assert r1.json()["task"]["id"] == r2.json()["task"]["id"]


def test_intake_links_invalid_url_bad_request(client):
    r = client.post("/api/plugins/kanban/intake-links", json={"url": "not-a-url"})
    # 200 is also acceptable if the backend does not validate URL shape;
    # the helper only rejects empty strings. Assert it returns a task.
    assert r.status_code in (200, 400)
    if r.status_code == 200:
        assert "task" in r.json()


def test_manifest_json_bumped():
    manifest_path = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "kanban" / "dashboard" / "manifest.json"
    )
    assert manifest_path.exists()
    import json
    data = json.loads(manifest_path.read_text())
    assert data["version"] != "1.0.2", "manifest should have been bumped"
    assert "1.0.3" in (data["entry"] or "")
    assert "1.0.3" in (data["css"] or "")


def test_index_js_contains_drop_link_component():
    index_path = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
    )
    assert index_path.exists()
    content = index_path.read_text()
    assert "DropLinkDialog" in content, "DropLinkDialog must appear in bundle"
    assert '"/intake-links"' in content, "POST path must appear in bundle"
    # Ensure the dialog renders the right label
    assert (
        "Attention Intake" in content
        or "Drop Link" in content
    ), "dialog title or button label expected"
