"""Tests for acp_adapter.terminal_bridge (ACP editor terminal routing)."""

from __future__ import annotations

import asyncio
import json
import threading
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from acp_adapter import terminal_bridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Caps:
    def __init__(self, terminal: bool = True):
        self.terminal = terminal


@dataclass
class _FakeClient:
    """Minimal async ACP client with scriptable terminal behavior."""

    exit_code: Optional[int] = 0
    signal: Optional[str] = None
    output: str = "hello\n"
    truncated: bool = False
    wait_delay: float = 0.0
    create_raises: Optional[Exception] = None
    calls: List[str] = field(default_factory=list)
    kill_event: threading.Event = field(default_factory=threading.Event)

    async def create_terminal(self, *, command, session_id, cwd=None,
                              args=None, env=None, output_byte_limit=None,
                              **kwargs):
        self.calls.append("create")
        if self.create_raises:
            raise self.create_raises
        self.last_command = command
        self.last_cwd = cwd
        return types.SimpleNamespace(terminal_id="term-1")

    async def wait_for_terminal_exit(self, *, session_id, terminal_id, **kw):
        self.calls.append("wait")
        if self.wait_delay:
            try:
                await asyncio.sleep(self.wait_delay)
            except asyncio.CancelledError:
                raise
        return types.SimpleNamespace(exit_code=self.exit_code, signal=self.signal)

    async def terminal_output(self, *, session_id, terminal_id, **kw):
        self.calls.append("output")
        return types.SimpleNamespace(output=self.output, truncated=self.truncated,
                                     exit_status=None)

    async def kill_terminal(self, *, session_id, terminal_id, **kw):
        self.calls.append("kill")
        self.kill_event.set()
        return None

    async def release_terminal(self, *, session_id, terminal_id, **kw):
        self.calls.append("release")
        return None


@pytest.fixture()
def loop_thread():
    """A real event loop running in a background thread (like the ACP server)."""

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _bind(client, loop, cwd, caps=None, monkeypatch=None, mode=None):
    if monkeypatch is not None:
        if mode is None:
            monkeypatch.delenv("HERMES_ACP_TERMINAL", raising=False)
        else:
            monkeypatch.setenv("HERMES_ACP_TERMINAL", mode)
    return terminal_bridge.use_acp_terminal(
        client=client,
        session_id="sess-1",
        loop=loop,
        cwd=cwd,
        capabilities=caps or _Caps(),
    )


REMOTE_CWD = "/workspace/definitely-not-a-local-dir-xyz"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_inactive_without_context():
    assert terminal_bridge.acp_terminal_active() is False
    assert terminal_bridge.run_command("true", timeout=5) is None


def test_auto_enables_for_remote_cwd(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        assert terminal_bridge.acp_terminal_active() is True


def test_auto_disables_for_local_cwd(tmp_path, loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, str(tmp_path), monkeypatch=monkeypatch):
        assert terminal_bridge.acp_terminal_active() is False
        assert terminal_bridge.run_command("true", timeout=5) is None


def test_always_enables_for_local_cwd(tmp_path, loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, str(tmp_path), monkeypatch=monkeypatch, mode="always"):
        assert terminal_bridge.acp_terminal_active() is True


def test_never_disables_remote(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch, mode="never"):
        assert terminal_bridge.acp_terminal_active() is False


def test_no_terminal_capability_disables(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, caps=_Caps(terminal=False),
               monkeypatch=monkeypatch):
        assert terminal_bridge.acp_terminal_active() is False


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def test_successful_command(loop_thread, monkeypatch):
    client = _FakeClient(exit_code=0, output="ok\n")
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("echo ok", timeout=10)
    assert result is not None
    assert result["exit_code"] == 0
    assert result["status"] == "success"
    assert "ok" in result["output"]
    assert result["backend"] == "acp-editor"
    assert client.calls == ["create", "wait", "output", "release"]
    assert client.last_command == "echo ok"
    assert client.last_cwd == REMOTE_CWD


def test_workdir_overrides_session_cwd(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        terminal_bridge.run_command("ls", timeout=10, workdir="/workspace/sub")
    assert client.last_cwd == "/workspace/sub"


def test_nonzero_exit(loop_thread, monkeypatch):
    client = _FakeClient(exit_code=2, output="boom\n")
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("false", timeout=10)
    assert result["exit_code"] == 2
    assert result["status"] == "error"
    assert "code 2" in result["error"]


def test_signal_exit(loop_thread, monkeypatch):
    client = _FakeClient(exit_code=None, signal="SIGKILL")
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("sleep 100", timeout=10)
    assert result["exit_code"] == -1
    assert result["signal"] == "SIGKILL"
    assert result["status"] == "error"


def test_background_rejected(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("sleep 100", timeout=10, background=True)
    assert result is not None
    assert result["status"] == "error"
    assert "background" in result["error"]
    # No terminal was ever created for a rejected background request.
    assert client.calls == []


def test_timeout_kills_and_salvages_output(loop_thread, monkeypatch):
    client = _FakeClient(wait_delay=60.0, output="partial...\n")
    monkeypatch.setattr(terminal_bridge, "_TIMEOUT_GRACE_SECONDS", 0.2)
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("sleep 100", timeout=0)
    assert result["status"] == "error"
    assert "timed out" in result["error"]
    assert "partial" in result["output"]
    assert "kill" in client.calls
    assert "release" in client.calls


def test_transport_failure_reports_not_run_locally(loop_thread, monkeypatch):
    client = _FakeClient(create_raises=RuntimeError("connection lost"))
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("echo hi", timeout=10)
    assert result["status"] == "error"
    assert "NOT run locally" in result["error"]


def test_output_truncation_marker(loop_thread, monkeypatch):
    client = _FakeClient(output="x" * 150_000)
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("cat big", timeout=10)
    assert result["truncated"] is True
    assert result["output"].startswith("[...output truncated")
    assert len(result["output"]) < 150_000


def test_pty_note(loop_thread, monkeypatch):
    client = _FakeClient()
    with _bind(client, loop_thread, REMOTE_CWD, monkeypatch=monkeypatch):
        result = terminal_bridge.run_command("top", timeout=10, pty=True)
    assert "pty" in result.get("note", "")


# ---------------------------------------------------------------------------
# Filesystem remote-mode integration
# ---------------------------------------------------------------------------

def test_filesystem_remote_workspace_flag(loop_thread):
    from acp_adapter import filesystem as acp_filesystem

    class _FsCaps:
        class fs:  # noqa: N801 — mimic pydantic attr shape
            read_text_file = True
            write_text_file = True

    class _FsClient:
        async def read_text_file(self, **kw):
            raise RuntimeError("Resource not found")

        async def write_text_file(self, **kw):
            raise RuntimeError("Resource not found")

    with acp_filesystem.use_acp_filesystem(
        client=_FsClient(),
        session_id="sess-1",
        loop=loop_thread,
        cwd=REMOTE_CWD,
        capabilities=_FsCaps(),
    ):
        assert acp_filesystem.acp_remote_workspace_active() is True
        # In remote mode a resource-miss must NOT fall back to local disk
        # (None) — it must surface as an explicit error.
        read_result = acp_filesystem.read_text_file("/workspace/app.py")
        assert read_result is not None
        assert read_result.error
        write_result = acp_filesystem.write_text_file("/workspace/app.py", "x")
        assert write_result is not None
        assert write_result.error


def test_filesystem_local_workspace_still_falls_back(tmp_path, loop_thread):
    from acp_adapter import filesystem as acp_filesystem

    class _FsCaps:
        class fs:  # noqa: N801
            read_text_file = True
            write_text_file = True

    class _FsClient:
        async def read_text_file(self, **kw):
            raise RuntimeError("Resource not found")

    with acp_filesystem.use_acp_filesystem(
        client=_FsClient(),
        session_id="sess-1",
        loop=loop_thread,
        cwd=str(tmp_path),
        capabilities=_FsCaps(),
    ):
        assert acp_filesystem.acp_remote_workspace_active() is False
        # Local workspace: resource-miss still falls back (returns None).
        assert acp_filesystem.read_text_file(str(tmp_path / "f.txt")) is None
