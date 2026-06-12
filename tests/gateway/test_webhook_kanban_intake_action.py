"""Tests for deterministic webhook ingestion into Attention Intake.

The ``kanban_intake_links`` webhook action is intentionally not an agent
prompt: it reuses the same Kanban helper as the dashboard Drop Link button and
must not call ``handle_message`` or incur an LLM run.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from hermes_cli import kanban_db as kb


def _make_adapter(routes) -> WebhookAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 0,
            "routes": routes,
        },
    )
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    kb.create_board("attention-intake")
    return home


@pytest.mark.asyncio
async def test_plain_text_newline_list_ingests_without_agent(kanban_home):
    routes = {
        "link-drop": {
            "secret": _INSECURE_NO_AUTH,
            "action": "kanban_intake_links",
        }
    }
    adapter = _make_adapter(routes)
    adapter.handle_message = AsyncMock()
    app = _create_app(adapter)

    body = "https://example.com/one\n\nhttps://example.com/two\n"
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/link-drop",
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "text/plain",
                "X-GitHub-Delivery": "plain-1",
            },
        )
        assert resp.status == 200
        data = await resp.json()

    assert data["status"] == "ingested"
    assert data["action"] == "kanban_intake_links"
    assert data["board"] == "attention-intake"
    assert data["count"] == 2
    assert [item["url"] for item in data["tasks"]] == [
        "https://example.com/one",
        "https://example.com/two",
    ]
    adapter.handle_message.assert_not_called()

    with kb.connect(board="attention-intake") as conn:
        for item in data["tasks"]:
            row = kb.get_task(conn, item["task_id"])
            assert row is not None
            assert row.status == "triage"
            assert row.assignee == "link-analyst"
            assert item["url"] in row.body
            assert "Attention Intake link-drop path" in row.body

    register_path = kanban_home / "artifacts" / "attention-intake" / "register.jsonl"
    register_lines = register_path.read_text(encoding="utf-8").splitlines()
    assert len(register_lines) == 2


@pytest.mark.asyncio
async def test_json_array_duplicate_reuses_existing_task(kanban_home):
    routes = {
        "links": {
            "secret": _INSECURE_NO_AUTH,
            "action": "kanban-intake-links",  # alias normalises in the handler
        }
    }
    adapter = _make_adapter(routes)
    adapter.handle_message = AsyncMock()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            "/webhooks/links",
            json=["https://example.com/dup", "https://example.com/dup"],
            headers={"X-GitHub-Delivery": "json-array-1"},
        )
        assert first.status == 200
        first_data = await first.json()

        second = await cli.post(
            "/webhooks/links",
            json=["https://example.com/dup"],
            headers={"X-GitHub-Delivery": "json-array-2"},
        )
        assert second.status == 200
        second_data = await second.json()

    assert first_data["count"] == 2
    assert first_data["tasks"][0]["task_id"] == first_data["tasks"][1]["task_id"]
    assert second_data["tasks"][0]["task_id"] == first_data["tasks"][0]["task_id"]
    adapter.handle_message.assert_not_called()

    register_path = kanban_home / "artifacts" / "attention-intake" / "register.jsonl"
    # Link-helper idempotency should keep duplicate drops from appending duplicate
    # provisional register rows.
    assert len(register_path.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.asyncio
async def test_json_string_can_contain_newline_delimited_links(kanban_home):
    routes = {
        "links": {
            "secret": _INSECURE_NO_AUTH,
            "action": "kanban_intake_links",
        }
    }
    adapter = _make_adapter(routes)
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/links",
            data=json.dumps("https://example.com/a\nhttps://example.com/b").encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Delivery": "json-string-1",
            },
        )
        assert resp.status == 200
        data = await resp.json()

    assert data["count"] == 2
    assert [item["url"] for item in data["tasks"]] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


@pytest.mark.asyncio
async def test_empty_payload_rejected_before_agent(kanban_home):
    routes = {
        "links": {
            "secret": _INSECURE_NO_AUTH,
            "action": "kanban_intake_links",
        }
    }
    adapter = _make_adapter(routes)
    adapter.handle_message = AsyncMock()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/links",
            data=b"\n\n",
            headers={"X-GitHub-Delivery": "empty-1"},
        )
        assert resp.status == 400
        data = await resp.json()

    assert "At least one link" in data["error"]
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_action_route_still_enforces_hmac(kanban_home):
    routes = {
        "links": {
            "secret": "real-secret",
            "action": "kanban_intake_links",
        }
    }
    adapter = _make_adapter(routes)
    adapter.handle_message = AsyncMock()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/webhooks/links",
            data=b"https://example.com/secure",
            headers={"X-GitHub-Delivery": "secure-1"},
        )
        assert resp.status == 401

    adapter.handle_message.assert_not_called()
    with kb.connect(board="attention-intake") as conn:
        assert kb.list_tasks(conn) == []
