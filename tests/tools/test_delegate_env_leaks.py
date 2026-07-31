"""Run-scoped env isolation for delegate_task children.

ENV-1/ENV-2 regression suite: a delegated subagent must not inherit the
parent dispatcher's HERMES_KANBAN_* identity or HERMES_INFERENCE_* launch
seed, and the parent process env must remain unchanged while children run.
"""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    child_env_lookup,
    delegated_child_context,
    delegated_child_subprocess_env,
    is_delegated_child_context,
)
from tools.delegate_tool import _run_single_child


# Run-scoped vars that must never leak into a child.
_RUN_SCOPED_VARS = [
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BRANCH",
    "HERMES_INFERENCE_MODEL",
    "HERMES_INFERENCE_PROVIDER",
]


def _make_parent_agent():
    """Minimal parent-like object for _run_single_child."""
    parent = MagicMock()
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._delegate_depth = 0
    parent._delegate_spinner = None
    parent.tool_progress_callback = None
    parent._current_task_id = None
    parent._touch_activity = lambda desc: None
    parent._current_turn_id = ""
    parent.session_id = "parent-session"
    parent.model = "parent/model"
    parent.provider = "parent-provider"
    parent.base_url = "http://localhost:1"
    parent.api_key = "parent-key"
    parent.api_mode = "chat_completions"
    parent.platform = "cli"
    parent.enabled_toolsets = ["terminal"]
    parent.iteration_budget = MagicMock()
    parent.iteration_budget.remaining = 100
    parent._client_kwargs = {"api_key": "***", "base_url": "http://localhost:1"}
    return parent


def _make_child_agent(parent):
    """Build a real AIAgent and patch away network/prompt setup."""
    from run_agent import AIAgent

    with patch("run_agent.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Done", tool_calls=None, refusal=None))],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        mock_client.close = MagicMock()
        MockOpenAI.return_value = mock_client
        with patch.object(AIAgent, "_build_system_prompt", return_value="You are a test agent"), \
             patch("agent.context_compressor.get_model_context_length", return_value=128_000):
            child = AIAgent(
                base_url=parent.base_url,
                api_key=parent.api_key,
                model=parent.model,
                provider=parent.provider,
                api_mode=parent.api_mode,
                max_iterations=3,
                enabled_toolsets=["terminal"],
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="subagent",
            )
    child._delegate_depth = 1
    child._subagent_id = "sa-0-test"
    child._delegate_role = "leaf"
    child._subagent_goal = "test goal"
    return child


@pytest.fixture
def seeded_env(monkeypatch):
    """Seed all run-scoped vars in the parent environment."""
    for name in _RUN_SCOPED_VARS:
        monkeypatch.setenv(name, f"parent-{name.lower().replace('hermes_', '').replace('_', '-')}")
    # Sanity: profile/workspace anchors stay present (not scrubbed).
    monkeypatch.setenv("HERMES_HOME", os.environ.get("HERMES_HOME", "/tmp/hermes-test"))
    monkeypatch.setenv("HERMES_PROFILE", "test-profile")


@pytest.mark.parametrize("var", _RUN_SCOPED_VARS)
def test_child_thread_env_is_scrubbed(seeded_env, monkeypatch, var):
    """A delegated child thread must not see any run-scoped env var."""
    captured = {}
    started = threading.Event()

    parent = _make_parent_agent()
    child = _make_child_agent(parent)
    parent._active_children.append(child)

    def patched_run(*args, **kwargs):
        started.set()
        captured["inside_child"] = {
            "lookup": child_env_lookup(var),
            "subprocess_env": delegated_child_subprocess_env(),
            "delegated": is_delegated_child_context(),
        }
        return {"final_response": "ok", "completed": True, "api_calls": 0, "messages": []}

    with patch.object(child, "run_conversation", patched_run):
        result = _run_single_child(0, "env probe", child, parent)

    assert result is not None
    assert result.get("status") in ("completed", "failed")
    assert started.is_set(), "child run_conversation was never entered"

    assert captured["inside_child"]["lookup"] is None, f"{var} leaked via child_env_lookup"
    assert captured["inside_child"]["delegated"] is True, "child context not marked delegated"
    subprocess_env = captured["inside_child"]["subprocess_env"]
    assert subprocess_env is not None
    assert var not in subprocess_env, f"{var} leaked into subprocess env"
    assert subprocess_env.get(DELEGATED_CHILD_ENV_MARKER) == "1"


def test_parent_env_unchanged_during_child_run(seeded_env, monkeypatch):
    """While a child is active, the parent's process env must not be mutated."""
    parent = _make_parent_agent()
    child = _make_child_agent(parent)
    parent._active_children.append(child)

    hold = threading.Event()
    parent_snapshot_at_start = {k: os.environ[k] for k in os.environ}
    snapshots = {}

    def patched_run(*args, **kwargs):
        snapshots["during_lookup"] = {var: child_env_lookup(var) for var in _RUN_SCOPED_VARS}
        snapshots["during_subprocess_env"] = delegated_child_subprocess_env()
        hold.wait(timeout=5)
        return {"final_response": "ok", "completed": True, "api_calls": 0, "messages": []}

    with patch.object(child, "run_conversation", patched_run):
        runner = threading.Thread(
            target=lambda: _run_single_child(0, "hold", child, parent),
            daemon=True,
        )
        runner.start()
        # Wait until the child has entered run_conversation and taken its snapshot.
        deadline = time.monotonic() + 3
        while snapshots.get("during_lookup") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        # While the child is still active, assert parent env is unchanged.
        assert dict(os.environ) == parent_snapshot_at_start, (
            "parent os.environ changed while child was active"
        )
        hold.set()
        runner.join(timeout=5)

    # After child finishes, parent env must still equal the original snapshot.
    assert {k: os.environ[k] for k in os.environ} == parent_snapshot_at_start, (
        "parent os.environ changed after child finished"
    )
    # The snapshot taken inside the child worker must differ on run-scoped keys.
    during_lookup = snapshots["during_lookup"]
    assert during_lookup is not None
    for var in _RUN_SCOPED_VARS:
        assert during_lookup[var] is None, f"child saw {var} during run"
    subprocess_env = snapshots["during_subprocess_env"]
    assert subprocess_env is not None
    for var in _RUN_SCOPED_VARS:
        assert var not in subprocess_env, f"{var} leaked into subprocess env during run"
    assert subprocess_env.get(DELEGATED_CHILD_ENV_MARKER) == "1"


def test_two_concurrent_children_do_not_leak_or_corrupt_parent(seeded_env):
    """Two concurrent children: neither sees run-scoped vars; parent env stays intact."""
    parent = _make_parent_agent()
    children = []
    for i in range(2):
        child = _make_child_agent(parent)
        child._subagent_id = f"sa-{i}-test"
        child._delegate_depth = 1
        children.append(child)
        parent._active_children.append(child)

    child_envs = {0: {}, 1: {}}
    gates = [threading.Event(), threading.Event()]

    def make_patched_run(idx):
        def patched_run(*args, **kwargs):
            child_envs[idx]["lookup"] = {var: child_env_lookup(var) for var in _RUN_SCOPED_VARS}
            child_envs[idx]["subprocess_env"] = delegated_child_subprocess_env()
            gates[idx].set()
            # Wait for the other child to also be active (controlled interleaving).
            other = 1 - idx
            gates[other].wait(timeout=5)
            return {"final_response": f"child-{idx}", "completed": True, "api_calls": 0, "messages": []}

        return patched_run

    parent_snapshot = {k: os.environ[k] for k in os.environ}

    threads = []
    patches = [
        patch.object(children[0], "run_conversation", make_patched_run(0)),
        patch.object(children[1], "run_conversation", make_patched_run(1)),
    ]
    for p in patches:
        p.start()
    try:
        for i, child in enumerate(children):
            t = threading.Thread(
                target=_run_single_child,
                args=(i, f"task-{i}", child, parent),
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join(timeout=10)
    finally:
        for p in patches:
            p.stop()

    final_parent = {k: os.environ[k] for k in os.environ}
    assert final_parent == parent_snapshot, "parent env was corrupted by concurrent children"

    for i, envs in child_envs.items():
        lookup = envs["lookup"]
        for var in _RUN_SCOPED_VARS:
            assert lookup[var] is None, f"child {i} inherited {var} via lookup"
        subprocess_env = envs["subprocess_env"]
        assert subprocess_env is not None
        for var in _RUN_SCOPED_VARS:
            assert var not in subprocess_env, f"child {i} inherited {var} via subprocess env"
        assert subprocess_env.get(DELEGATED_CHILD_ENV_MARKER) == "1"


def test_background_delegation_leaves_parent_env_untouched(seeded_env):
    """A child running on a background executor must not mutate the parent env."""
    from tools.daemon_pool import DaemonThreadPoolExecutor

    parent = _make_parent_agent()
    child = _make_child_agent(parent)
    child._subagent_id = "sa-bg-0"
    parent._active_children.append(child)

    hold = threading.Event()
    captured = {}
    parent_snapshot = {k: os.environ[k] for k in os.environ}

    def patched_run(*args, **kwargs):
        captured["child_lookup"] = {var: child_env_lookup(var) for var in _RUN_SCOPED_VARS}
        captured["child_subprocess_env"] = delegated_child_subprocess_env()
        hold.wait(timeout=5)
        return {"final_response": "bg done", "completed": True, "api_calls": 0, "messages": []}

    with patch.object(child, "run_conversation", patched_run):
        future = DaemonThreadPoolExecutor(max_workers=1).submit(
            _run_single_child, 0, "background env probe", child, parent
        )
        # Parent continues running while child is active.
        deadline = time.monotonic() + 2
        while not captured and time.monotonic() < deadline:
            time.sleep(0.05)
        assert captured, "background child never started"
        # Parent env must remain unchanged while child runs.
        assert {k: os.environ[k] for k in os.environ} == parent_snapshot, (
            "parent env changed while background child ran"
        )
        hold.set()
        future.result(timeout=5)

    assert {k: os.environ[k] for k in os.environ} == parent_snapshot, (
        "parent env changed after background child finished"
    )
    for var in _RUN_SCOPED_VARS:
        assert captured["child_lookup"][var] is None, f"background child inherited {var} via lookup"
    subprocess_env = captured["child_subprocess_env"]
    assert subprocess_env is not None
    for var in _RUN_SCOPED_VARS:
        assert var not in subprocess_env, f"background child inherited {var} via subprocess env"
    assert subprocess_env.get(DELEGATED_CHILD_ENV_MARKER) == "1"


def test_hermes_kanban_branch_is_scrubbed(seeded_env, monkeypatch):
    """HERMES_KANBAN_BRANCH is in the scrub set and must not leak."""
    parent = _make_parent_agent()
    child = _make_child_agent(parent)
    parent._active_children.append(child)

    monkeypatch.setenv("HERMES_KANBAN_BRANCH", "feature/kanban-branch")
    captured = {}

    def patched_run(*args, **kwargs):
        captured["lookup"] = child_env_lookup("HERMES_KANBAN_BRANCH")
        captured["subprocess_env"] = delegated_child_subprocess_env()
        return {"final_response": "ok", "completed": True, "api_calls": 0, "messages": []}

    with patch.object(child, "run_conversation", patched_run):
        _run_single_child(0, "branch probe", child, parent)

    assert captured["lookup"] is None, "HERMES_KANBAN_BRANCH leaked via child_env_lookup"
    assert "HERMES_KANBAN_BRANCH" not in captured["subprocess_env"], (
        "HERMES_KANBAN_BRANCH leaked into subprocess env"
    )


def test_child_env_overlay_used_for_in_process_lookup(seeded_env):
    """child_env_lookup respects the overlay inside a delegated child context."""
    from agent.delegation_context import delegated_child_context

    assert child_env_lookup("HERMES_KANBAN_TASK") == os.environ.get("HERMES_KANBAN_TASK")

    with delegated_child_context(overlay={"HERMES_KANBAN_TASK": None}):
        assert child_env_lookup("HERMES_KANBAN_TASK") is None

    with delegated_child_context(overlay={"HERMES_KANBAN_TASK": "shadow"}):
        assert child_env_lookup("HERMES_KANBAN_TASK") == "shadow"

    assert child_env_lookup("HERMES_KANBAN_TASK") == os.environ.get("HERMES_KANBAN_TASK")


def test_delegated_child_does_not_activate_kanban_stop_nudge(seeded_env, monkeypatch):
    """A delegated child must not be treated as the parent Kanban worker.

    Even when the parent os.environ still carries HERMES_KANBAN_TASK, the
    overlay-aware resolver inside the child context must return None for the
    worker identity, and the stop-guard must be disabled.
    """
    from agent.kanban_stop import build_kanban_stop_nudge, kanban_stop_nudge_enabled

    # Outside child context the parent env is visible, so the guard would fire.
    assert os.environ.get("HERMES_KANBAN_TASK")
    assert kanban_stop_nudge_enabled() is True

    with delegated_child_context(overlay={"HERMES_KANBAN_TASK": None}):
        # The child must not see the parent's task identity.
        assert child_env_lookup("HERMES_KANBAN_TASK") is None
        # Therefore the stop-guard is off.
        assert kanban_stop_nudge_enabled() is False
        # And no nudge message is produced.
        assert build_kanban_stop_nudge(messages=[]) is None

    # Parent env must remain intact after the child context exits.
    assert os.environ.get("HERMES_KANBAN_TASK")
    assert kanban_stop_nudge_enabled() is True


def test_delegated_child_budget_exhaustion_does_not_record_parent_failure(
    seeded_env, tmp_path, monkeypatch,
):
    """Budget exhaustion in a delegated child must not mutate the parent's task/run.

    The DB mutation layer already rejects delegated children via
    ``_assert_not_delegated_child_mutation``, but ``finalize_turn`` must also
    fail fast so it does not attempt to record a timeout against the parent.
    """
    from unittest.mock import MagicMock, patch

    from agent.delegation_context import delegated_child_context
    from agent.turn_finalizer import finalize_turn
    from hermes_cli import kanban_db as _kb

    # Set up a real temp kanban board so any mistaken mutation would have
    # somewhere to land.
    db_path = tmp_path / "test_kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    _kb.init_db(db_path)
    parent_tid = _kb.create_task(
        _kb.connect(),
        title="parent task",
        assignee="platform-eng",
    )
    # Move the task to running with a known run id.
    conn = _kb.connect()
    claimed = _kb.claim_task(conn, parent_tid, ttl_seconds=3600, claimer="test-claimer")
    assert claimed is not None, "claim failed"
    parent_run_id = claimed.current_run_id

    # Prepare a fake agent with the budget-exhausted state.
    agent = MagicMock()
    agent.max_iterations = 5
    agent.iteration_budget = MagicMock()
    agent.iteration_budget.remaining = 0
    agent.quiet_mode = True
    agent._handle_max_iterations = MagicMock(return_value="summary")
    agent._skill_nudge_interval = -1  # disabled for this test
    agent._iters_since_skill = 0
    agent._turn_completion_explainer_enabled = MagicMock(return_value=False)
    agent._file_mutation_verifier_enabled = MagicMock(return_value=False)

    record_spy = MagicMock()

    # Seed the parent env into the process, but run finalization inside a
    # delegated child overlay.  The child must not record failure.
    monkeypatch.setenv("HERMES_KANBAN_TASK", parent_tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(parent_run_id))

    with patch.object(_kb, "_record_task_failure", record_spy):
        with delegated_child_context(overlay={
            "HERMES_KANBAN_TASK": None,
            "HERMES_KANBAN_RUN_ID": None,
        }):
            result = finalize_turn(
                agent,
                final_response=None,
                api_call_count=5,
                interrupted=False,
                failed=False,
                messages=[],
                conversation_history=[],
                effective_task_id="",
                turn_id="",
                user_message="test",
                original_user_message="test",
                _should_review_memory=False,
                _turn_exit_reason="budget_exhausted",
            )

    # No failure should have been recorded for the parent task.
    assert record_spy.call_count == 0, "delegated child recorded failure against parent task"

    # The parent task must still be running and unmutated.
    conn = _kb.connect()
    task = _kb.get_task(conn, parent_tid)
    assert task is not None
    assert task.status == "running"
    assert task.current_run_id == parent_run_id

    # Sanity: outside delegated context, the same conditions DO record failure.
    calls = []
    real_record = _kb._record_task_failure
    def recording_record(*args, **kwargs):
        calls.append((args, kwargs))
        return real_record(*args, **kwargs)

    with patch.object(_kb, "_record_task_failure", side_effect=recording_record):
        result = finalize_turn(
            agent,
            final_response=None,
            api_call_count=5,
            interrupted=False,
            failed=False,
            messages=[],
            conversation_history=[],
            effective_task_id="",
            turn_id="",
            user_message="test",
            original_user_message="test",
            _should_review_memory=False,
            _turn_exit_reason="budget_exhausted",
        )

    assert len(calls) == 1, "non-delegated worker did not record failure"
    args, kwargs = calls[0]
    assert args[1] == parent_tid
    assert kwargs.get("outcome") == "timed_out"
    assert kwargs.get("expected_run_id") == parent_run_id

    # And the parent task was indeed transitioned away from running.
    task = _kb.get_task(_kb.connect(), parent_tid)
    assert task is not None
    assert task.status in {"ready", "blocked"}
    assert task.current_run_id != parent_run_id


def test_delegated_child_inference_provider_resolution(seeded_env, tmp_path, monkeypatch):
    """A runtime-provider consumer must resolve HERMES_INFERENCE_PROVIDER through the overlay.

    This is a semantic repro: it exercises ``resolve_requested_provider``,
    an actual model/provider selection consumer, not just the
    ``child_env_lookup`` helper.
    """
    from hermes_cli.runtime_provider import resolve_requested_provider

    # Parent env is seeded by the fixture; no config provider is set.
    assert child_env_lookup("HERMES_INFERENCE_PROVIDER")
    assert resolve_requested_provider() == child_env_lookup("HERMES_INFERENCE_PROVIDER")

    with delegated_child_context(overlay={"HERMES_INFERENCE_PROVIDER": "child-provider"}):
        assert resolve_requested_provider() == "child-provider"

    # Parent env remains intact after the child exits.
    assert child_env_lookup("HERMES_INFERENCE_PROVIDER") == os.environ.get("HERMES_INFERENCE_PROVIDER")
    assert resolve_requested_provider() == os.environ.get("HERMES_INFERENCE_PROVIDER")


def test_env3_db_mutation_fails_closed_for_delegated_child(seeded_env, tmp_path, monkeypatch):
    """ENV-3's _record_task_failure CAS is blocked at the DB layer by write_txn.

    The in-process guard in finalize_turn is a fast-fail optimization; this
    test proves the DB mutation layer independently rejects delegated children,
    so a bug that bypassed the finalizer guard still could not corrupt the
    parent's task/run.
    """
    from hermes_cli import kanban_db as _kb

    db_path = tmp_path / "test_kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    _kb.init_db(db_path)
    parent_tid = _kb.create_task(
        _kb.connect(),
        title="parent task",
        assignee="platform-eng",
    )
    conn = _kb.connect()
    claimed = _kb.claim_task(conn, parent_tid, ttl_seconds=3600, claimer="test-claimer")
    assert claimed is not None
    parent_run_id = claimed.current_run_id

    with pytest.raises(PermissionError):
        with delegated_child_context(overlay={
            "HERMES_KANBAN_TASK": None,
            "HERMES_KANBAN_RUN_ID": None,
        }):
            _kb._record_task_failure(
                _kb.connect(),
                parent_tid,
                error="should not land",
                outcome="timed_out",
                expected_run_id=parent_run_id,
            )

    # Parent task is untouched.
    task = _kb.get_task(_kb.connect(), parent_tid)
    assert task is not None
    assert task.status == "running"
    assert task.current_run_id == parent_run_id
