"""Turn-end guard for kanban workers.

Kanban workers must end with ``kanban_complete`` or ``kanban_block``. Models
(especially GLM / Qwen families) sometimes narrate the next step
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({"kanban_complete", "kanban_block"})
_PROGRESS_KANBAN_TOOL = "kanban_progress"

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_progress_enabled() -> bool:
    """Return whether semantic Kanban checkpoints are enabled (default off)."""
    env = os.environ.get("HERMES_KANBAN_PROGRESS")
    if env is not None:
        normalized = env.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    try:
        from hermes_cli.config import cfg_get, load_config

        return bool(
            cfg_get(load_config(), "agent", "kanban_progress_enabled", default=False)
        )
    except Exception:
        return False


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def _current_user_turn(messages: Iterable[dict] | None) -> list[dict]:
    """Return messages at/after the latest user boundary.

    Goal-mode continuations reuse the durable transcript. A checkpoint from an
    earlier user turn must not authorize a later narrated exit.
    """
    items = list(messages or [])
    for index in range(len(items) - 1, -1, -1):
        msg = items[index]
        if isinstance(msg, dict) and msg.get("role") == "user":
            return items[index:]
    return items


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def session_ended_kanban_turn_legitimately(
    messages: Iterable[dict] | None,
) -> bool:
    """True after a terminal call, or an enabled semantic checkpoint."""
    if session_called_kanban_terminal(messages):
        return True
    if not kanban_progress_enabled():
        return False
    for msg in _current_user_turn(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            if any(
                _tool_call_name(tc) == _PROGRESS_KANBAN_TOOL
                for tc in msg.get("tool_calls") or []
            ):
                return True
        elif (
            msg.get("role") == "tool"
            and str(msg.get("name") or "") == _PROGRESS_KANBAN_TOOL
        ):
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a follow-up when a worker exits without a legitimate board tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked/checkpointed, or nudge budget exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_ended_kanban_turn_legitimately(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    progress_enabled = kanban_progress_enabled()
    progress_option = (
        ", OR `kanban_progress(summary=..., metadata=...)` if real work was "
        "done but the task is continuing"
        if progress_enabled
        else ""
    )
    terminal_tools = "`kanban_complete` / `kanban_block`"
    if progress_enabled:
        terminal_tools += " / enabled checkpoint"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        f"{terminal_tools}).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked"
        f"{progress_option}.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_progress_enabled",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "session_ended_kanban_turn_legitimately",
]
