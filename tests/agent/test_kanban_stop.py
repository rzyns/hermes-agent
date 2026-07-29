"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch, tmp_path):
    for var in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_STOP_NUDGE",
        "HERMES_KANBAN_PROGRESS",
        "HERMES_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    # Keep config-backed feature flags hermetic. Deleting HERMES_HOME alone
    # falls back to the developer's real default profile, so an enabled local
    # checkpoint flag can make the default-off byte-contract test flaky.
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return monkeypatch


def test_disabled_without_kanban_task(clear_kanban_env):
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_enabled_with_kanban_task(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert kanban_stop_nudge_enabled() is True


def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_progress_flag_off_preserves_pre_feature_nudge_bytes(clear_kanban_env):
    """Default-OFF must preserve the exact pre-kanban_progress wire text."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    expected = (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        "Task `t_46be8aa5` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )

    actual = build_kanban_stop_nudge(messages=[], attempts=0)

    assert actual is not None
    assert actual.encode("utf-8") == expected.encode("utf-8")


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_no_nudge_after_kanban_block(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {"role": "tool", "name": "kanban_block", "tool_call_id": "1", "content": "blocked"},
    ]
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_budget_exhausted(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    assert build_kanban_stop_nudge(messages=[], attempts=2) is None
    assert build_kanban_stop_nudge(messages=[], attempts=1, max_attempts=1) is None
    assert build_kanban_stop_nudge(messages=[], attempts=0, max_attempts=1) is not None


# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


def test_nudge_text_warns_about_blocking(clear_kanban_env):
    """The nudge should warn that repeated violations will block the task."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    nudge = build_kanban_stop_nudge(messages=[], attempts=0)
    assert nudge is not None
    assert "block" in nudge.lower(), (
        "nudge should warn that repeated violations will block the task"
    )


def test_nudge_and_dispatcher_budgets_are_independent(clear_kanban_env):
    """Agent-side nudge budget (2) and dispatcher-side streak (3) are
    separate budgets — the nudge counter does not affect the dispatcher's
    violation streak, and vice versa.

    This is a source-level invariant check: the nudge counter
    (``_kanban_stop_nudges``) lives on the AIAgent instance and resets
    per session, while the dispatcher streak lives in the task_runs DB
    table and persists across worker respawns.
    """
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    # Agent-side: 2 nudge attempts per session
    assert build_kanban_stop_nudge(messages=[], attempts=0) is not None
    assert build_kanban_stop_nudge(messages=[], attempts=1) is not None
    assert build_kanban_stop_nudge(messages=[], attempts=2) is None
    # Dispatcher-side streak is tracked in the DB, not in the nudge module —
    # the nudge module has no knowledge of the streak counter.
    assert not hasattr(build_kanban_stop_nudge, "_streak")


def test_progress_suppresses_nudge_only_when_config_enabled(
    clear_kanban_env, tmp_path
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_HOME", str(tmp_path))
    messages = [{"role": "tool", "name": "kanban_progress", "content": "ok"}]

    assert build_kanban_stop_nudge(messages=messages) is not None

    (tmp_path / "config.yaml").write_text(
        "agent:\n  kanban_progress_enabled: true\n", encoding="utf-8"
    )
    assert build_kanban_stop_nudge(messages=messages) is None


def test_progress_env_override_enables_checkpoint(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_PROGRESS", "1")
    messages = [{"role": "tool", "name": "kanban_progress", "content": "ok"}]

    assert build_kanban_stop_nudge(messages=messages) is None


def test_progress_only_satisfies_the_current_user_turn(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_PROGRESS", "1")
    messages = [
        {"role": "user", "content": "first turn"},
        {"role": "tool", "name": "kanban_progress", "content": "ok"},
        {"role": "user", "content": "continue the goal"},
        {"role": "assistant", "content": "I will continue later"},
    ]

    assert build_kanban_stop_nudge(messages=messages) is not None


def test_progress_enabled_nudge_names_checkpoint_without_changing_budget(
    clear_kanban_env, tmp_path
):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "agent:\n  kanban_progress_enabled: true\n", encoding="utf-8"
    )

    nudge = build_kanban_stop_nudge(messages=[], attempts=1)
    assert nudge is not None
    assert "kanban_progress" in nudge
    assert build_kanban_stop_nudge(messages=[], attempts=2) is None
