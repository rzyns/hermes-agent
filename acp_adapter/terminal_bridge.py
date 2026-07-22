"""ACP editor terminal bridge.

Runs Hermes ``terminal`` commands in the *editor's* workspace via ACP client
``terminal/*`` requests, instead of a local execution environment.

This matters when the ACP client is an editor whose workspace lives on a
machine Hermes cannot reach directly — e.g. VS Code Remote Development
(Remote-SSH, Dev Containers, attached Kubernetes containers) with Hermes
running on the user's local machine. File reads/writes already route through
the client (see ``acp_adapter.filesystem``); without this module, terminal
commands would still execute locally against a workspace cwd that does not
exist here.

Mirrors the design of ``acp_adapter.filesystem``: the ACP SDK methods are
async while Hermes tools run synchronously inside the ACP executor thread, so
the active client/loop/session are bound via a ContextVar and awaited with
``asyncio.run_coroutine_threadsafe``.

Routing policy (``HERMES_ACP_TERMINAL`` env var):

- ``auto`` (default): route through the editor only when the session cwd
  does not exist locally — i.e. the workspace is remote to Hermes. Local
  editor workspaces keep the full-featured local execution path (background
  processes, pty, containers).
- ``always``: route whenever the client advertises the terminal capability.
- ``never``: never route.

Limitations (v1): ``background=True``, ``pty=True`` and watch/notify flags
are not supported on the editor path — callers get a clear error for
background requests instead of a silent local run against a bogus cwd.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import contextvars
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Client-side retention limit for command output (bytes).
_OUTPUT_BYTE_LIMIT = 1_000_000
# Model-facing output cap (characters), matching the spirit of the local
# bounded-capture path. Truncated from the head (tail is kept).
_MODEL_OUTPUT_CHAR_LIMIT = 100_000
# Extra seconds past the command timeout before we give up on the client.
_TIMEOUT_GRACE_SECONDS = 10.0
# How long to wait for output/release after killing a timed-out command.
_POST_KILL_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class ACPTerminalContext:
    """Active ACP terminal-routing context for the synchronous terminal tool."""

    client: Any
    session_id: str
    loop: asyncio.AbstractEventLoop
    cwd: str | None = None
    enabled: bool = False
    remote_workspace: bool = False


_context: contextvars.ContextVar[ACPTerminalContext | None] = contextvars.ContextVar(
    "acp_terminal_context",
    default=None,
)


def supports_terminal(capabilities: Any) -> bool:
    """Return whether ACP client capabilities include terminal/*."""

    return bool(getattr(capabilities, "terminal", False))


def is_remote_workspace(cwd: str | None) -> bool:
    """A session cwd that does not exist locally marks a remote workspace."""

    if not cwd:
        return False
    try:
        return not os.path.isdir(cwd)
    except OSError:
        return True


def _routing_mode() -> str:
    mode = (os.environ.get("HERMES_ACP_TERMINAL") or "auto").strip().lower()
    if mode not in {"auto", "always", "never"}:
        logger.warning("Unknown HERMES_ACP_TERMINAL=%r; using 'auto'", mode)
        return "auto"
    return mode


@contextlib.contextmanager
def use_acp_terminal(
    *,
    client: Any,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    cwd: str | None,
    capabilities: Any,
) -> Iterator[None]:
    """Bind ACP editor terminal routing for tools in this context."""

    mode = _routing_mode()
    remote = is_remote_workspace(cwd)
    if not supports_terminal(capabilities) or mode == "never":
        enabled = False
    elif mode == "always":
        enabled = True
    else:  # auto
        enabled = remote

    ctx = ACPTerminalContext(
        client=client,
        session_id=session_id,
        loop=loop,
        cwd=cwd,
        enabled=enabled,
        remote_workspace=remote,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> ACPTerminalContext | None:
    """Return the currently bound ACP terminal context, if any."""

    return _context.get()


def acp_terminal_active() -> bool:
    """True when terminal commands should route through the ACP client."""

    ctx = current_context()
    return bool(ctx and ctx.enabled)


def _run_client_coro(ctx: ACPTerminalContext, coro: Any, timeout: float) -> Any:
    """Run an ACP client coroutine from the synchronous tool thread."""

    future = asyncio.run_coroutine_threadsafe(coro, ctx.loop)
    return future.result(timeout=timeout)


def _truncate_for_model(output: str, client_truncated: bool) -> tuple[str, bool]:
    truncated = bool(client_truncated)
    if len(output) > _MODEL_OUTPUT_CHAR_LIMIT:
        output = output[-_MODEL_OUTPUT_CHAR_LIMIT:]
        truncated = True
    if truncated:
        output = (
            "[...output truncated; earlier output was dropped...]\n" + output
        )
    return output, truncated


def _collect_output(ctx: ACPTerminalContext, terminal_id: str, timeout: float) -> tuple[str, bool]:
    response = _run_client_coro(
        ctx,
        ctx.client.terminal_output(session_id=ctx.session_id, terminal_id=terminal_id),
        timeout,
    )
    output = getattr(response, "output", "")
    if not isinstance(output, str):
        output = str(output or "")
    return output, bool(getattr(response, "truncated", False))


def _release_quietly(ctx: ACPTerminalContext, terminal_id: str) -> None:
    try:
        _run_client_coro(
            ctx,
            ctx.client.release_terminal(session_id=ctx.session_id, terminal_id=terminal_id),
            _POST_KILL_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.debug("ACP terminal release failed for %s", terminal_id, exc_info=True)


def run_command(
    command: str,
    *,
    timeout: int,
    workdir: Optional[str] = None,
    background: bool = False,
    pty: bool = False,
) -> Optional[dict]:
    """Run ``command`` in the editor's workspace via ACP terminal/*.

    Returns a dict shaped like the ``terminal`` tool's JSON payload
    (``output`` / ``exit_code`` / ``error`` / ``status``), or ``None`` when
    routing is not active so the caller falls through to local execution.
    """

    ctx = current_context()
    if ctx is None or not ctx.enabled:
        return None

    if background:
        return {
            "output": "",
            "exit_code": -1,
            "error": (
                "background=true is not supported while this session's "
                "workspace lives on the editor's remote host (commands run "
                "through the editor over ACP). Re-run in the foreground — "
                "long tasks can use nohup/redirection plus a follow-up "
                "status command, e.g. `nohup cmd > /tmp/task.log 2>&1 &` "
                "then `tail /tmp/task.log`."
            ),
            "status": "error",
            "backend": "acp-editor",
        }

    cwd = workdir or ctx.cwd
    notes: list[str] = []
    if pty:
        notes.append("pty=true is ignored on the ACP editor terminal path.")

    logger.info(
        "ACP terminal routing: %r (cwd=%s, timeout=%ss)",
        command if len(command) <= 200 else command[:197] + "...",
        cwd,
        timeout,
    )

    terminal_id: str | None = None
    try:
        create_response = _run_client_coro(
            ctx,
            ctx.client.create_terminal(
                command=command,
                session_id=ctx.session_id,
                cwd=cwd,
                output_byte_limit=_OUTPUT_BYTE_LIMIT,
            ),
            _POST_KILL_TIMEOUT_SECONDS + 20.0,
        )
        terminal_id = getattr(create_response, "terminal_id", None)
        if not terminal_id:
            return {
                "output": "",
                "exit_code": -1,
                "error": "ACP client did not return a terminal id.",
                "status": "error",
                "backend": "acp-editor",
            }

        try:
            exit_response = _run_client_coro(
                ctx,
                ctx.client.wait_for_terminal_exit(
                    session_id=ctx.session_id, terminal_id=terminal_id
                ),
                float(timeout) + _TIMEOUT_GRACE_SECONDS,
            )
        except concurrent.futures.TimeoutError:
            # Timed out: kill, salvage partial output, release.
            with contextlib.suppress(Exception):
                _run_client_coro(
                    ctx,
                    ctx.client.kill_terminal(
                        session_id=ctx.session_id, terminal_id=terminal_id
                    ),
                    _POST_KILL_TIMEOUT_SECONDS,
                )
            partial, client_truncated = "", False
            with contextlib.suppress(Exception):
                partial, client_truncated = _collect_output(
                    ctx, terminal_id, _POST_KILL_TIMEOUT_SECONDS
                )
            _release_quietly(ctx, terminal_id)
            terminal_id = None
            partial, _ = _truncate_for_model(partial, client_truncated)
            return {
                "output": partial,
                "exit_code": -1,
                "error": (
                    f"Command timed out after {timeout}s in the editor "
                    "workspace and was killed."
                ),
                "status": "error",
                "backend": "acp-editor",
            }

        output, client_truncated = _collect_output(
            ctx, terminal_id, _POST_KILL_TIMEOUT_SECONDS
        )
        _release_quietly(ctx, terminal_id)
        terminal_id = None

        exit_code = getattr(exit_response, "exit_code", None)
        signal = getattr(exit_response, "signal", None)
        if exit_code is None:
            exit_code = 0 if not signal else -1

        output, truncated = _truncate_for_model(output, client_truncated)
        result: dict[str, Any] = {
            "output": output,
            "exit_code": exit_code,
            "error": "" if exit_code == 0 else f"Command exited with code {exit_code}",
            "status": "success" if exit_code == 0 else "error",
            "backend": "acp-editor",
        }
        if signal:
            result["signal"] = signal
            if not result["error"]:
                result["error"] = f"Command terminated by signal {signal}"
                result["status"] = "error"
        if truncated:
            result["truncated"] = True
        if notes:
            result["note"] = " ".join(notes)
        return result
    except Exception as exc:  # noqa: BLE001 — surface any transport failure
        if terminal_id:
            _release_quietly(ctx, terminal_id)
        logger.warning("ACP editor terminal failed: %s", exc, exc_info=True)
        return {
            "output": "",
            "exit_code": -1,
            "error": (
                f"ACP editor terminal failed: {exc}. The workspace is on the "
                "editor's host; the command was NOT run locally."
            ),
            "status": "error",
            "backend": "acp-editor",
        }
