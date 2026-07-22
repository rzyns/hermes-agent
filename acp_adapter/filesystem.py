"""ACP editor filesystem bridge.

This module lets synchronous Hermes file tools call ACP client filesystem
requests while an ACP session is running.  The ACP SDK methods are async, while
Hermes tools run synchronously inside the ACP executor thread, so the active
client/loop/session are bound via a ContextVar and awaited with
``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from tools.file_operations import PatchResult, ReadResult, WriteResult, normalize_read_pagination


@dataclass(frozen=True)
class ACPFilesystemContext:
    """Active ACP filesystem context for synchronous file tools."""

    client: Any
    session_id: str
    loop: asyncio.AbstractEventLoop
    cwd: str | None = None
    can_read: bool = False
    can_write: bool = False
    timeout: float = 30.0
    # True when the session cwd does not exist on this machine — i.e. the
    # editor's workspace is remote to Hermes (VS Code Remote-SSH / containers
    # / Kubernetes). In that mode the local-disk fallback is disabled: a
    # "resource not found" from the client must surface as an error instead
    # of silently reading/writing a same-named local path.
    remote_workspace: bool = False


_context: contextvars.ContextVar[ACPFilesystemContext | None] = contextvars.ContextVar(
    "acp_filesystem_context",
    default=None,
)


def _cap_bool(capabilities: Any, name: str) -> bool:
    fs = getattr(capabilities, "fs", None)
    return bool(getattr(fs, name, False))


def supports_read(capabilities: Any) -> bool:
    """Return whether ACP client capabilities include fs/read_text_file."""

    return _cap_bool(capabilities, "read_text_file")


def supports_write(capabilities: Any) -> bool:
    """Return whether ACP client capabilities include fs/write_text_file."""

    return _cap_bool(capabilities, "write_text_file")


@contextlib.contextmanager
def use_acp_filesystem(
    *,
    client: Any,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    cwd: str | None,
    capabilities: Any,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Bind ACP editor filesystem access for file tools in this context."""

    remote_workspace = False
    if cwd:
        try:
            remote_workspace = not os.path.isdir(cwd)
        except OSError:
            remote_workspace = True
    ctx = ACPFilesystemContext(
        client=client,
        session_id=session_id,
        loop=loop,
        cwd=cwd,
        can_read=supports_read(capabilities),
        can_write=supports_write(capabilities),
        timeout=timeout,
        remote_workspace=remote_workspace,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> ACPFilesystemContext | None:
    """Return the currently bound ACP filesystem context, if any."""

    return _context.get()


def _absolute_path(path: str, cwd: str | None) -> str:
    raw = os.path.expanduser(str(path or ""))
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(cwd or os.getcwd()) / candidate
    return str(candidate.resolve(strict=False))


def _run_client_coro(ctx: ACPFilesystemContext, coro: Any) -> Any:
    """Run an ACP client coroutine from the synchronous tool thread."""

    future = asyncio.run_coroutine_threadsafe(coro, ctx.loop)
    return future.result(timeout=ctx.timeout)


def _add_line_numbers(content: str, offset: int) -> str:
    lines = content.splitlines()
    return "\n".join(f"{offset + idx:6d}|{line}" for idx, line in enumerate(lines))


def acp_read_active() -> bool:
    """Return True when file tools are running with ACP read support."""

    ctx = current_context()
    return bool(ctx and ctx.can_read)


def acp_remote_workspace_active() -> bool:
    """True when the active ACP session's workspace is remote to Hermes."""

    ctx = current_context()
    return bool(ctx and ctx.remote_workspace)


def _should_fallback_to_local_filesystem(exc: Exception) -> bool:
    """Return True when the ACP editor filesystem could not handle the path.

    Some clients advertise fs/read_text_file or fs/write_text_file for dirty
    buffer access but only serve files they have materialized in the editor
    resource layer. Zed can also return a generic internal error for writes to
    paths it does not own. In those cases, normal local disk reads/writes should
    still work via the existing fallback path.
    """

    message = str(exc).lower()
    return "resource not found" in message or "internal error" in message


def read_text_file(path: str, offset: int = 1, limit: int = 500) -> ReadResult | None:
    """Read through ACP fs/read_text_file if active and supported.

    Returns ``None`` when there is no active ACP filesystem or the client did
    not advertise read support, allowing callers to fall back unchanged.
    """

    ctx = current_context()
    if ctx is None or not ctx.can_read:
        return None
    offset, limit = normalize_read_pagination(offset, limit)
    abs_path = _absolute_path(path, ctx.cwd)
    try:
        response = _run_client_coro(
            ctx,
            ctx.client.read_text_file(
                path=abs_path,
                session_id=ctx.session_id,
                limit=limit,
                line=offset,
            ),
        )
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content or "")
        line_count = len(content.splitlines())
        return ReadResult(
            content=_add_line_numbers(content, offset),
            total_lines=(offset + line_count - 1) if line_count else 0,
            file_size=len(content.encode("utf-8")),
            truncated=False,
        )
    except Exception as exc:
        if _should_fallback_to_local_filesystem(exc) and not ctx.remote_workspace:
            return None
        return ReadResult(error=f"ACP editor filesystem read failed for '{abs_path}': {exc}")


def read_text_file_raw(path: str) -> ReadResult | None:
    """Read a whole file through ACP fs/read_text_file for patch operations."""

    ctx = current_context()
    if ctx is None or not ctx.can_read:
        return None
    abs_path = _absolute_path(path, ctx.cwd)
    try:
        response = _run_client_coro(
            ctx,
            ctx.client.read_text_file(
                path=abs_path,
                session_id=ctx.session_id,
                limit=1_000_000,
                line=1,
            ),
        )
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content or "")
        return ReadResult(
            content=content,
            total_lines=len(content.splitlines()),
            file_size=len(content.encode("utf-8")),
            truncated=False,
        )
    except Exception as exc:
        if _should_fallback_to_local_filesystem(exc) and not ctx.remote_workspace:
            return None
        return ReadResult(error=f"ACP editor filesystem read failed for '{abs_path}': {exc}")


def write_text_file(path: str, content: str) -> WriteResult | None:
    """Write through ACP fs/write_text_file if active and supported.

    Returns ``None`` when there is no active ACP filesystem or the client did
    not advertise write support, allowing callers to fall back unchanged.
    """

    ctx = current_context()
    if ctx is None or not ctx.can_write:
        return None
    abs_path = _absolute_path(path, ctx.cwd)
    try:
        _run_client_coro(
            ctx,
            ctx.client.write_text_file(
                content=content,
                path=abs_path,
                session_id=ctx.session_id,
            ),
        )
        return WriteResult(
            bytes_written=len(content.encode("utf-8")),
            dirs_created=False,
        )
    except Exception as exc:
        if _should_fallback_to_local_filesystem(exc) and not ctx.remote_workspace:
            return None
        return WriteResult(error=f"ACP editor filesystem write failed for '{abs_path}': {exc}")

class _ACPFileOperations:
    """Minimal file-operations adapter backed by ACP read/write requests."""

    def read_file_raw(self, path: str) -> ReadResult:
        result = read_text_file_raw(path)
        if result is None:
            return ReadResult(error="ACP editor filesystem is not available for this file")
        return result

    def write_file(self, path: str, content: str) -> WriteResult:
        result = write_text_file(path, content)
        if result is None:
            return WriteResult(error="ACP editor filesystem is not available for this file")
        return result

    def delete_file(self, path: str) -> WriteResult:
        return WriteResult(error="ACP editor filesystem does not support delete operations")

    def move_file(self, src: str, dst: str) -> WriteResult:
        return WriteResult(error="ACP editor filesystem does not support move operations")


def _unified_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def patch_replace(path: str, old_string: str, new_string: str, replace_all: bool = False) -> PatchResult | None:
    """Apply replace-mode patch through ACP read/write when available."""

    ctx = current_context()
    if ctx is None or not (ctx.can_read and ctx.can_write):
        return None
    read_result = read_text_file_raw(path)
    if read_result is None:
        return None
    if read_result.error:
        return PatchResult(error=read_result.error)

    from tools.fuzzy_match import fuzzy_find_and_replace

    content = read_result.content or ""
    new_content, match_count, _strategy, error = fuzzy_find_and_replace(
        content, old_string, new_string, replace_all
    )
    if error or match_count == 0:
        err_msg = error or f"Could not find match for old_string in {path}"
        try:
            from tools.fuzzy_match import format_no_match_hint

            err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
        except Exception:
            pass
        return PatchResult(error=err_msg)

    write_result = write_text_file(path, new_content)
    if write_result is None:
        return None
    if write_result.error:
        return PatchResult(error=f"Failed to write changes: {write_result.error}")

    return PatchResult(
        success=True,
        diff=_unified_diff(path, content, new_content),
        files_modified=[path],
        lsp_diagnostics=write_result.lsp_diagnostics,
    )


def patch_v4a(
    patch_content: str,
    resolved_paths: Mapping[str, str | None] | None = None,
) -> PatchResult | None:
    """Apply V4A patch updates/adds through ACP read/write when available."""

    ctx = current_context()
    if ctx is None or not (ctx.can_read and ctx.can_write):
        return None

    from tools.patch_parser import OperationType, apply_v4a_operations, parse_v4a_patch

    operations, parse_error = parse_v4a_patch(patch_content)
    if parse_error:
        return PatchResult(error=f"Failed to parse patch: {parse_error}")
    if resolved_paths:
        for op in operations:
            resolved_file = resolved_paths.get(op.file_path)
            if resolved_file:
                op.file_path = resolved_file
            if op.new_path:
                resolved_new = resolved_paths.get(op.new_path)
                if resolved_new:
                    op.new_path = resolved_new
    if any(op.operation in {OperationType.DELETE, OperationType.MOVE} for op in operations):
        if ctx.remote_workspace:
            return PatchResult(error=(
                "Delete/Move patch operations are not supported through the "
                "ACP editor filesystem, and this session's workspace is on "
                "the editor's remote host (no local fallback). Use the "
                "terminal tool (rm / mv) instead."
            ))
        return None
    if len(operations) != 1:
        # ACP editor writes are not transactional. Avoid partially applying a
        # multi-file patch if a later editor write reports a resource miss;
        # file_tools will retry the whole patch through its local fallback with
        # the same resolved paths.
        if ctx.remote_workspace:
            return PatchResult(error=(
                "Multi-file patches are not supported through the ACP editor "
                "filesystem (writes are not transactional), and this "
                "session's workspace is on the editor's remote host (no "
                "local fallback). Apply the patch one file at a time."
            ))
        return None

    op = operations[0]
    if op.operation is OperationType.ADD:
        # A single Add File can be attempted atomically; if the client reports a
        # resource miss/internal error, return None so file_tools uses the
        # normal local filesystem fallback.
        content_lines = [
            line.content
            for hunk in op.hunks
            for line in hunk.lines
            if line.prefix == "+"
        ]
        content = "\n".join(content_lines)
        write_result = write_text_file(op.file_path, content)
        if write_result is None:
            return None
        if write_result.error:
            return PatchResult(error=f"Failed to add {op.file_path}: {write_result.error}")
        diff = f"--- /dev/null\n+++ b/{op.file_path}\n"
        diff += "\n".join(f"+{line}" for line in content_lines)
        return PatchResult(
            success=True,
            diff=diff,
            files_created=[op.file_path],
            lsp_diagnostics=write_result.lsp_diagnostics,
        )

    # For update patches, prove the editor can serve the existing target
    # before entering the two-phase patch helper. If the ACP client reports a
    # resource miss/internal error, return None so file_tools can use the normal
    # local filesystem fallback without surfacing an ACP-only validation error.
    for op in operations:
        if op.operation is OperationType.UPDATE:
            read_result = read_text_file_raw(op.file_path)
            if read_result is None:
                return None
            if read_result.error:
                return PatchResult(error=read_result.error)

    result = apply_v4a_operations(operations, _ACPFileOperations())
    if result.error and "ACP editor filesystem is not available for this file" in result.error:
        return None
    return result
