"""Tests for the SSE run-events proxy endpoints in web_server.py.

The dashboard (FastAPI, port 9119) proxies ``/api/runs/{run_id}/events`` to
the api_server adapter (aiohttp, port 8642).  These tests exercise:

1. The resolver helpers (``_resolve_api_server_base_url``,
   ``_resolve_api_server_key``).
2. The SSE proxy's graceful-degradation path (synthetic error frame when the
   api_server is unreachable).
3. The capabilities proxy's graceful-degradation path.
4. The ``_require_token_or_query`` auth helper (query token acceptance for
   EventSource, which cannot set headers).
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


@pytest.fixture
def proxy_client(monkeypatch, _isolate_hermes_home):
    """TestClient with auth disabled (loopback mode)."""
    previous_auth = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "test-session-token")
    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        if previous_auth is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth


class TestResolveApiServerBaseUrl:
    """``_resolve_api_server_base_url`` reads env/config defaults."""

    def test_defaults_when_no_config(self, monkeypatch):
        monkeypatch.delenv("API_SERVER_HOST", raising=False)
        monkeypatch.delenv("API_SERVER_PORT", raising=False)
        url = web_server._resolve_api_server_base_url()
        # Falls back to adapter defaults 127.0.0.1:8642
        assert "127.0.0.1" in url
        assert "8642" in url

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_HOST", "10.0.0.5")
        monkeypatch.setenv("API_SERVER_PORT", "9999")
        url = web_server._resolve_api_server_base_url()
        assert url == "http://10.0.0.5:9999"

    def test_config_yaml_host_port(self, monkeypatch, tmp_path):
        monkeypatch.delenv("API_SERVER_HOST", raising=False)
        monkeypatch.delenv("API_SERVER_PORT", raising=False)
        cfg_file = tmp_path / ".hermes" / "config.yaml"
        cfg_file.parent.mkdir(parents=True)
        cfg_file.write_text(
            "platforms:\n  api_server:\n    host: 192.168.1.10\n    port: 7777\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        url = web_server._resolve_api_server_base_url()
        assert "192.168.1.10" in url
        assert "7777" in url


class TestResolveApiServerKey:
    """``_resolve_api_server_key`` reads the scoped/env key."""

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_KEY", "sk-test-key-123456789012345")
        key = web_server._resolve_api_server_key()
        assert key == "sk-test-key-123456789012345"

    def test_no_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        key = web_server._resolve_api_server_key()
        assert key == ""


class TestRequireTokenOrQuery:
    """``_require_token_or_query`` accepts both header and query token."""

    def test_header_token_accepted(self, proxy_client):
        """Valid session header authenticates."""
        resp = proxy_client.get(
            "/api/runs/test-run/events",
            headers={"X-Hermes-Session-Token": "test-session-token"},
        )
        # The response will be a streaming response or an error frame;
        # we just need to confirm it's NOT a 401.
        assert resp.status_code != 401

    def test_query_token_accepted(self, proxy_client):
        """Valid ?token= query param authenticates (EventSource path)."""
        resp = proxy_client.get(
            f"/api/runs/test-run/events?{urlencode({'token': 'test-session-token'})}",
        )
        assert resp.status_code != 401

    def test_no_token_rejected(self, proxy_client):
        """No token at all → 401."""
        resp = proxy_client.get("/api/runs/test-run/events")
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, proxy_client):
        """Wrong token → 401."""
        resp = proxy_client.get(
            f"/api/runs/test-run/events?{urlencode({'token': 'wrong-token'})}",
        )
        assert resp.status_code == 401


class TestSseProxyGracefulDegradation:
    """When the api_server is unreachable, the proxy degrades gracefully."""

    def test_events_proxy_emits_error_frame(self, proxy_client, monkeypatch):
        """When the upstream is unreachable, the SSE proxy emits a synthetic
        ``hermes.run_events.proxy_error`` frame instead of crashing."""
        # Point at a port nothing is listening on so httpx.ConnectError fires.
        monkeypatch.setenv("API_SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("API_SERVER_PORT", "1")  # port 1 = nothing listening

        resp = proxy_client.get(
            "/api/runs/test-run/events",
            headers={"X-Hermes-Session-Token": "test-session-token"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "hermes.run_events.proxy_error" in body
        # The frame should be parseable as SSE.
        assert "data:" in body

    def test_capabilities_proxy_returns_unavailable(self, proxy_client, monkeypatch):
        """When the upstream is unreachable, the capabilities endpoint returns
        ``{"available": false}`` with a reason."""
        monkeypatch.setenv("API_SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("API_SERVER_PORT", "1")

        resp = proxy_client.get(
            "/api/runs/capabilities",
            headers={"X-Hermes-Session-Token": "test-session-token"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert "reason" in data


class TestSseProxyUrlConstruction:
    """The proxy correctly constructs the upstream URL and forwards params."""

    def test_run_events_unavailable_bytes_shape(self):
        """``_run_events_unavailable_bytes`` produces a well-formed SSE frame."""
        frame = web_server._run_events_unavailable_bytes("test reason")
        text = frame.decode("utf-8")
        assert "event: hermes.run_events.proxy_error" in text
        assert '"reason": "test reason"' in text
        assert text.endswith("\n\n")

    def test_run_events_unavailable_bytes_json_valid(self):
        """The data payload is valid JSON."""
        frame = web_server._run_events_unavailable_bytes("x")
        text = frame.decode("utf-8")
        # Extract the data line.
        for line in text.split("\n"):
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                assert payload["proxy_error"] is True
                assert payload["reason"] == "x"
                return
        pytest.fail("No data: line found in frame")
