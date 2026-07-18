"""Tests for ACP editor filesystem dirty-buffer integration."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from acp.schema import ClientCapabilities, FileSystemCapabilities

from acp_adapter import filesystem as acp_filesystem
from acp_adapter.server import HermesACPAgent
from tools import terminal_tool
from tools.file_tools import patch_tool, read_file_tool, write_file_tool


class FakeACPClient:
    def __init__(
        self,
        *,
        read_content: str = "dirty\nbuffer\n",
        fail: Exception | None = None,
        read_fail: Exception | None = None,
        write_fail: Exception | None = None,
    ):
        self.read_content = read_content
        self.read_fail = read_fail if read_fail is not None else fail
        self.write_fail = write_fail if write_fail is not None else fail
        self.read_calls: list[dict] = []
        self.write_calls: list[dict] = []

    async def read_text_file(self, **kwargs):
        self.read_calls.append(kwargs)
        if self.read_fail is not None:
            raise self.read_fail
        return SimpleNamespace(content=self.read_content)

    async def write_text_file(self, **kwargs):
        self.write_calls.append(kwargs)
        if self.write_fail is not None:
            raise self.write_fail
        return None


def _caps(*, read: bool = False, write: bool = False) -> ClientCapabilities:
    return ClientCapabilities(
        fs=FileSystemCapabilities(readTextFile=read, writeTextFile=write)
    )


async def _with_acp_context(fn, *, client, session_id, cwd, capabilities):
    loop = asyncio.get_running_loop()

    def run_in_tool_thread():
        with acp_filesystem.use_acp_filesystem(
            client=client,
            session_id=session_id,
            loop=loop,
            cwd=str(cwd),
            capabilities=capabilities,
        ):
            return fn()

    return await asyncio.to_thread(run_in_tool_thread)


@pytest.fixture
def record_task_cwd():
    recorded_task_ids: list[str] = []

    def record(task_id: str, cwd) -> None:
        terminal_tool.record_session_cwd(task_id, str(cwd))
        recorded_task_ids.append(task_id)

    yield record

    for task_id in recorded_task_ids:
        terminal_tool.clear_session_cwd(task_id)


@pytest.mark.asyncio
async def test_dirty_buffer_read_uses_acp_client(tmp_path, record_task_cwd):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk\n", encoding="utf-8")
    client = FakeACPClient(read_content="dirty buffer\nsecond line\n")
    task_id = f"acp-fs-read-{uuid.uuid4()}"
    record_task_cwd(task_id, tmp_path)

    raw = await _with_acp_context(
        lambda: read_file_tool("example.txt", offset=1, limit=5, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True),
    )

    payload = json.loads(raw)
    assert "dirty buffer" in payload["content"]
    assert "clean disk" not in payload["content"]
    assert client.read_calls == [
        {
            "path": str(disk_file),
            "session_id": "session-1",
            "limit": 5,
            "line": 1,
        }
    ]


@pytest.mark.asyncio
async def test_dirty_buffer_read_uses_live_task_cwd_before_acp_cwd(
    tmp_path, record_task_cwd
):
    acp_root = tmp_path / "editor-root"
    task_cwd = tmp_path / "terminal-cwd"
    acp_root.mkdir()
    task_cwd.mkdir()
    client = FakeACPClient(read_content="dirty task cwd\n")
    task_id = f"acp-fs-read-live-cwd-{uuid.uuid4()}"
    record_task_cwd(task_id, task_cwd)

    raw = await _with_acp_context(
        lambda: read_file_tool("example.txt", offset=1, limit=5, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=acp_root,
        capabilities=_caps(read=True),
    )

    payload = json.loads(raw)
    assert "dirty task cwd" in payload["content"]
    assert client.read_calls[0]["path"] == str(task_cwd / "example.txt")


def test_no_capability_read_falls_back_to_local_disk(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk\n", encoding="utf-8")
    client = FakeACPClient(read_content="dirty buffer\n")
    task_id = f"acp-fs-fallback-{uuid.uuid4()}"

    async def run():
        return await _with_acp_context(
            lambda: read_file_tool(str(disk_file), task_id=task_id),
            client=client,
            session_id="session-1",
            cwd=tmp_path,
            capabilities=_caps(read=False),
        )

    raw = asyncio.run(run())
    payload = json.loads(raw)
    assert "clean disk" in payload["content"]
    assert client.read_calls == []


@pytest.mark.asyncio
async def test_write_uses_acp_client_without_local_disk_double_mutation(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("original disk\n", encoding="utf-8")
    client = FakeACPClient()
    task_id = f"acp-fs-write-{uuid.uuid4()}"

    raw = await _with_acp_context(
        lambda: write_file_tool(str(disk_file), "editor content\n", task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(write=True),
    )

    payload = json.loads(raw)
    assert payload["bytes_written"] == len("editor content\n".encode("utf-8"))
    assert "warning" not in payload
    assert disk_file.read_text(encoding="utf-8") == "original disk\n"
    assert client.write_calls == [
        {
            "content": "editor content\n",
            "path": str(disk_file),
            "session_id": "session-1",
        }
    ]


@pytest.mark.asyncio
async def test_editor_resource_not_found_falls_back_to_local_disk(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk\n", encoding="utf-8")
    client = FakeACPClient(fail=RuntimeError("Resource not found"))
    task_id = f"acp-fs-resource-miss-{uuid.uuid4()}"

    raw = await _with_acp_context(
        lambda: read_file_tool(str(disk_file), task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True),
    )

    payload = json.loads(raw)
    assert "clean disk" in payload["content"]
    assert "error" not in payload
    assert client.read_calls


@pytest.mark.asyncio
async def test_editor_write_internal_error_falls_back_to_local_disk(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("original disk\n", encoding="utf-8")
    client = FakeACPClient(fail=RuntimeError("Internal error"))
    task_id = f"acp-fs-write-fallback-{uuid.uuid4()}"

    raw = await _with_acp_context(
        lambda: write_file_tool(str(disk_file), "local fallback\n", task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(write=True),
    )

    payload = json.loads(raw)
    assert payload["bytes_written"] == len("local fallback\n".encode("utf-8"))
    assert "error" not in payload
    assert disk_file.read_text(encoding="utf-8") == "local fallback\n"
    assert client.write_calls


@pytest.mark.asyncio
async def test_repeated_acp_read_refetches_dirty_buffer_instead_of_deduping(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk\n", encoding="utf-8")
    client = FakeACPClient(read_content="dirty v1\n")
    task_id = f"acp-fs-read-dedup-{uuid.uuid4()}"

    async def run_once():
        return await _with_acp_context(
            lambda: read_file_tool(str(disk_file), offset=1, limit=5, task_id=task_id),
            client=client,
            session_id="session-1",
            cwd=tmp_path,
            capabilities=_caps(read=True),
        )

    first = json.loads(await run_once())
    client.read_content = "dirty v2\n"
    second = json.loads(await run_once())

    assert "dirty v1" in first["content"]
    assert "dirty v2" in second["content"]
    assert second.get("status") != "unchanged"
    assert len(client.read_calls) == 2


@pytest.mark.asyncio
async def test_patch_replace_uses_acp_dirty_buffer_and_write(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk old\n", encoding="utf-8")
    client = FakeACPClient(read_content="dirty buffer old\n")
    task_id = f"acp-fs-patch-replace-{uuid.uuid4()}"

    raw = await _with_acp_context(
        lambda: patch_tool(
            mode="replace",
            path=str(disk_file),
            old_string="dirty buffer old",
            new_string="dirty buffer new",
            task_id=task_id,
        ),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert "dirty buffer new" in payload["diff"]
    assert disk_file.read_text(encoding="utf-8") == "clean disk old\n"
    assert client.write_calls[-1]["content"] == "dirty buffer new\n"


@pytest.mark.asyncio
async def test_patch_v4a_uses_acp_dirty_buffer_and_write(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk old\n", encoding="utf-8")
    client = FakeACPClient(read_content="alpha\nold\nomega\n")
    task_id = f"acp-fs-patch-v4a-{uuid.uuid4()}"
    patch = """*** Begin Patch
*** Update File: example.txt
@@ old @@
 alpha
-old
+new
 omega
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert "new" in payload["diff"]
    assert disk_file.read_text(encoding="utf-8") == "clean disk old\n"
    assert client.write_calls[-1]["content"] == "alpha\nnew\nomega\n"


@pytest.mark.asyncio
async def test_patch_v4a_uses_resolved_live_task_path_for_acp(
    tmp_path, record_task_cwd
):
    acp_root = tmp_path / "editor-root"
    task_cwd = tmp_path / "terminal-cwd"
    acp_root.mkdir()
    task_cwd.mkdir()
    client = FakeACPClient(read_content="old\n")
    task_id = f"acp-fs-patch-v4a-live-cwd-{uuid.uuid4()}"
    target = task_cwd / "example.txt"
    record_task_cwd(task_id, task_cwd)
    patch = """*** Begin Patch
*** Update File: example.txt
@@ old @@
-old
+new
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=acp_root,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert {call["path"] for call in client.read_calls} == {str(target)}
    assert client.write_calls[-1]["path"] == str(target)
    assert payload["files_modified"] == [str(target.resolve())]


@pytest.mark.asyncio
async def test_patch_v4a_resource_not_found_falls_back_to_local_disk(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("alpha\nold\nomega\n", encoding="utf-8")
    client = FakeACPClient(fail=RuntimeError("Resource not found"))
    task_id = f"acp-fs-patch-v4a-fallback-{uuid.uuid4()}"
    patch = f"""*** Begin Patch
*** Update File: {disk_file}
@@ old @@
 alpha
-old
+local fallback
 omega
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert "local fallback" in disk_file.read_text(encoding="utf-8")
    assert client.read_calls


@pytest.mark.asyncio
async def test_patch_v4a_fallback_uses_resolved_live_task_path(
    tmp_path, record_task_cwd
):
    acp_root = tmp_path / "editor-root"
    task_cwd = tmp_path / "terminal-cwd"
    acp_root.mkdir()
    task_cwd.mkdir()
    task_target = task_cwd / "example.txt"
    acp_target = acp_root / "example.txt"
    task_target.write_text("old\n", encoding="utf-8")
    acp_target.write_text("old in editor root\n", encoding="utf-8")
    client = FakeACPClient(fail=RuntimeError("Resource not found"))
    task_id = f"acp-fs-patch-v4a-fallback-live-cwd-{uuid.uuid4()}"
    record_task_cwd(task_id, task_cwd)
    patch = """*** Begin Patch
*** Update File: example.txt
@@ old @@
-old
+new
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=acp_root,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert task_target.read_text(encoding="utf-8") == "new\n"
    assert acp_target.read_text(encoding="utf-8") == "old in editor root\n"
    assert payload["files_modified"] == [str(task_target.resolve())]


@pytest.mark.asyncio
async def test_patch_v4a_write_resource_not_found_falls_back_to_local_disk(
    tmp_path, record_task_cwd
):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("alpha\nold\nomega\n", encoding="utf-8")
    client = FakeACPClient(
        read_content="alpha\nold\nomega\n",
        write_fail=RuntimeError("Resource not found"),
    )
    task_id = f"acp-fs-patch-v4a-write-fallback-{uuid.uuid4()}"
    record_task_cwd(task_id, tmp_path)
    patch = """*** Begin Patch
*** Update File: example.txt
@@ old @@
 alpha
-old
+local write fallback
 omega
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert "local write fallback" in disk_file.read_text(encoding="utf-8")
    assert client.read_calls
    assert client.write_calls


@pytest.mark.asyncio
async def test_patch_v4a_add_uses_acp_write_without_local_disk(
    tmp_path, record_task_cwd
):
    disk_file = tmp_path / "created.txt"
    client = FakeACPClient()
    task_id = f"acp-fs-patch-v4a-add-{uuid.uuid4()}"
    record_task_cwd(task_id, tmp_path)
    patch = """*** Begin Patch
*** Add File: created.txt
+alpha
+beta
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert not disk_file.exists()
    assert client.write_calls == [
        {
            "content": "alpha\nbeta",
            "path": str(disk_file),
            "session_id": "session-1",
        }
    ]


@pytest.mark.asyncio
async def test_patch_v4a_add_resource_not_found_falls_back_to_local_disk(tmp_path):
    disk_file = tmp_path / "created.txt"
    client = FakeACPClient(fail=RuntimeError("Resource not found"))
    task_id = f"acp-fs-patch-v4a-add-fallback-{uuid.uuid4()}"
    patch = f"""*** Begin Patch
*** Add File: {disk_file}
+alpha
+beta
*** End Patch
"""

    raw = await _with_acp_context(
        lambda: patch_tool(mode="patch", patch=patch, task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True, write=True),
    )

    payload = json.loads(raw)
    assert payload["success"] is True
    assert disk_file.read_text(encoding="utf-8") == "alpha\nbeta"
    assert client.write_calls


@pytest.mark.asyncio
async def test_acp_failure_returns_clear_error_without_local_fallback(tmp_path):
    disk_file = tmp_path / "example.txt"
    disk_file.write_text("clean disk\n", encoding="utf-8")
    client = FakeACPClient(fail=RuntimeError("zed unavailable"))
    task_id = f"acp-fs-failure-{uuid.uuid4()}"

    raw = await _with_acp_context(
        lambda: read_file_tool(str(disk_file), task_id=task_id),
        client=client,
        session_id="session-1",
        cwd=tmp_path,
        capabilities=_caps(read=True),
    )

    payload = json.loads(raw)
    assert "ACP editor filesystem read failed" in payload["error"]
    assert "zed unavailable" in payload["error"]
    assert client.read_calls


@pytest.mark.asyncio
async def test_server_stores_client_filesystem_capabilities():
    agent = HermesACPAgent()

    await agent.initialize(client_capabilities=_caps(read=True, write=False))

    assert agent.client_supports_fs_read() is True
    assert agent.client_supports_fs_write() is False
