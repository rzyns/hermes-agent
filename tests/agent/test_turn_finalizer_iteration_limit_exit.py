"""Regression tests for iteration-limit exit normalization (#61631)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_finalizer import finalize_turn
from hermes_cli import kanban_db as kb


class _LimitAgent:
    def __init__(
        self,
        *,
        max_iterations=60,
        budget_remaining=0,
        completion_explainer=False,
    ):
        self.max_iterations = max_iterations
        self.iteration_budget = SimpleNamespace(
            remaining=budget_remaining, used=max_iterations, max_total=max_iterations
        )
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages = None
        self._handle_max_iterations_called = False
        self._completion_explainer = completion_explainer

    def _handle_max_iterations(self, messages, api_call_count):
        self._handle_max_iterations_called = True
        return "summary from extra call"

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return self._completion_explainer

    def _format_turn_completion_explanation(self, _reason):
        return "iteration-limit explanation"

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _finalize(
    agent,
    *,
    final_response,
    exit_reason,
    api_call_count=60,
    pending_verification_response=None,
):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response=pending_verification_response,
    )


def _claimed_worker(tmp_path, monkeypatch, **task_kwargs):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="boundary worker",
            assignee="worker",
            **task_kwargs,
        )
        kb.claim_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        assert run is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run.id))
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    return task_id, run.id


def _complete_worker(summary):
    from tools import kanban_tools as kt

    return json.loads(kt._handle_complete({"summary": summary}))


def test_successful_kanban_complete_at_iteration_boundary_survives_finalizer(
    tmp_path, monkeypatch
):
    task_id, _run_id = _claimed_worker(tmp_path, monkeypatch)
    completion = _complete_worker("finished on the final tool call")
    assert completion["ok"] is True

    _finalize(
        _LimitAgent(max_iterations=1),
        final_response=None,
        exit_reason="unknown",
        api_call_count=1,
    )

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        runs = kb.list_runs(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert task is not None
    assert task.status == "done"
    assert [run.outcome for run in runs] == ["completed"]
    assert not [event for event in events if event.kind in {"timed_out", "gave_up"}]


def test_post_finalizer_completion_requires_grace_for_same_latest_run(
    tmp_path, monkeypatch
):
    task_id, run_id = _claimed_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"terminal_completion_grace_seconds": 30}},
    )

    _finalize(
        _LimitAgent(max_iterations=1),
        final_response=None,
        exit_reason="unknown",
        api_call_count=1,
    )
    completion = _complete_worker("finished while finalizer benched the run")

    assert completion["ok"] is True
    assert completion["accepted_terminal_after_budget_bench"] is True
    assert completion["budget_run_id"] == run_id
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "done"


def test_post_finalizer_completion_grace_defaults_to_zero(tmp_path, monkeypatch):
    task_id, _run_id = _claimed_worker(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    assert kb._terminal_completion_grace_seconds() == 0

    _finalize(
        _LimitAgent(max_iterations=1),
        final_response=None,
        exit_reason="unknown",
        api_call_count=1,
    )
    completion = _complete_worker("too late without configured grace")

    assert "error" in completion
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready"


def test_grace_rejects_completion_after_newer_run_starts(tmp_path, monkeypatch):
    task_id, stale_run_id = _claimed_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"terminal_completion_grace_seconds": 30}},
    )
    _finalize(
        _LimitAgent(max_iterations=1),
        final_response=None,
        exit_reason="unknown",
        api_call_count=1,
    )
    with kb.connect() as conn:
        kb.claim_task(conn, task_id)
        newer_run = kb.latest_run(conn, task_id)
        assert newer_run is not None
    assert newer_run.id != stale_run_id

    completion = _complete_worker("stale completion after retry started")

    assert "error" in completion
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert task.current_run_id == newer_run.id


def test_worker_override_prevents_profile_budget_failure_on_first_tool_turn(
    tmp_path, monkeypatch
):
    captured = {}

    class _FakeProc:
        pid = 4246

    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kwargs: captured.update(cmd=cmd) or _FakeProc(),
    )
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "agent": {"max_turns": 1},
            "kanban": {"terminal_completion_grace_seconds": 0},
        },
    )
    task_id, _run_id = _claimed_worker(
        tmp_path,
        monkeypatch,
        goal_mode=True,
        goal_max_turns=30,
        worker_max_turns=4,
    )
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
    kb._default_spawn(task, str(tmp_path))
    worker_budget = int(captured["cmd"][captured["cmd"].index("--max-turns") + 1])

    _finalize(
        _LimitAgent(max_iterations=worker_budget, budget_remaining=3),
        final_response="tool turn complete",
        exit_reason="text_response(finish_reason=stop)",
        api_call_count=1,
    )

    from hermes_cli.config import load_config

    assert load_config()["agent"]["max_turns"] == 1
    assert task.goal_max_turns == 30
    assert captured["cmd"][captured["cmd"].index("--max-turns") + 1] == "4"
    assert worker_budget == 4
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert not [
        event
        for event in events
        if event.payload and "Iteration budget exhausted (1/1)" in str(event.payload)
    ]


def test_pending_verify_response_is_preserved_for_cron_delivery(monkeypatch):
    """A held-back verification response survives last-turn exhaustion."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "complete cron report body"

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response=report,
    )

    assert result["final_response"] == report
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is False


def test_pending_pre_verify_response_is_preserved_on_budget_exhaustion(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "budget exhausted but complete"

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="budget_exhausted",
        pending_verification_response=report,
    )

    assert result["final_response"] == report
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is False


def test_empty_pending_verification_response_uses_summary_fallback(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="",
    )

    assert result["final_response"] == "summary from extra call"
    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    assert agent._handle_max_iterations_called is True


def test_short_generated_summary_keeps_abnormal_turn_explainer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(completion_explainer=True)
    agent._handle_max_iterations = lambda *_args: "The"

    result = _finalize(agent, final_response=None, exit_reason="unknown")

    assert result["final_response"] == "The\n\niteration-limit explanation"


def test_short_preserved_verification_response_is_not_rewritten(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(completion_explainer=True)

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="The",
    )

    assert result["final_response"] == "The"


def test_text_response_exit_not_rewritten_at_iteration_limit(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent(budget_remaining=5)
    exit_reason = "text_response(finish_reason=stop)"

    result = _finalize(
        agent,
        final_response="normal answer",
        exit_reason=exit_reason,
        api_call_count=59,
    )

    assert result["turn_exit_reason"] == exit_reason
    assert agent._handle_max_iterations_called is False


@pytest.mark.parametrize(
    "exit_reason",
    [
        "error_near_max_iterations(boom)",
        "guardrail_halt",
        "partial_stream_recovery",
        "fallback_prior_turn_content",
        "empty_response_exhausted",
    ],
)
def test_unrelated_non_success_response_is_not_reclassified(monkeypatch, exit_reason):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response="diagnostic or partial content",
        exit_reason=exit_reason,
    )

    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


@pytest.mark.parametrize(
    ("exit_reason", "interrupted", "failed"),
    [
        ("interrupted_by_user", True, False),
        ("all_retries_exhausted_no_response", False, False),
        ("provider_failure", False, True),
    ],
)
def test_pending_response_does_not_mask_later_terminal_exit(
    monkeypatch, exit_reason, interrupted, failed
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()

    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=60,
        interrupted=interrupted,
        failed=failed,
        messages=[{"role": "user", "content": "task"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
        _pending_verification_response="stale premature report",
    )

    assert result["final_response"] is None
    assert result["turn_exit_reason"] == exit_reason
    assert result["completed"] is False
    assert agent._handle_max_iterations_called is False


def test_pending_response_records_kanban_timeout(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "456")
    record = MagicMock(name="record_task_failure")
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("hermes_cli.kanban_db.connect", lambda: conn)
    monkeypatch.setattr("hermes_cli.kanban_db._record_task_failure", record)
    agent = _LimitAgent()

    result = _finalize(
        agent,
        final_response=None,
        exit_reason="unknown",
        pending_verification_response="composed report",
    )

    assert result["turn_exit_reason"] == "max_iterations_reached(60/60)"
    record.assert_called_once_with(
        conn,
        "task-123",
        error=(
            "Iteration budget exhausted (60/60) — task could not complete "
            "within the allowed iterations"
        ),
        outcome="timed_out",
        release_claim=True,
        end_run=True,
        expected_run_id=456,
        event_payload_extra={"budget_used": 60, "budget_max": 60},
    )


def test_published_pending_candidate_is_not_duplicated_by_finalizer(monkeypatch):
    """When budget exhaustion preserves a verification candidate that is
    already the tail assistant message, the finalizer must NOT append a
    duplicate. The content-comparison guard prevents this. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "the composed report"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        # The candidate is already in messages as the tail assistant.
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The tail assistant already matches final_response — no duplicate appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # Persisted messages should also have no duplicate.
    assert agent.persisted_messages is not None
    persisted_roles = [m["role"] for m in agent.persisted_messages]
    assert persisted_roles == ["user", "assistant"]


def test_terminal_verification_failure_is_persisted_as_one_correction(monkeypatch):
    """When verification fails terminally (nudge present but budget exhausted),
    the finalizer drops the synthetic nudge and the assistant candidate
    persists as a single correction. No duplicate assistant appended. (#65919 §7)
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = _LimitAgent()
    report = "terminal failure correction"

    result = finalize_turn(
        agent,
        final_response=report,
        api_call_count=60,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": report},
            # Synthetic nudge — should be dropped by _drop_verification_continuation_scaffolding.
            {"role": "user", "content": "[System: run tests]", "_verification_stop_synthetic": True},
        ],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="task",
        original_user_message="task",
        _should_review_memory=False,
        _turn_exit_reason="unknown",
        _pending_verification_response=report,
    )

    # The nudge is dropped; the assistant candidate is the tail and matches
    # final_response, so no duplicate is appended.
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant"]
    # The nudge is gone from persisted messages too.
    assert agent.persisted_messages is not None
    persisted_contents = [m.get("content") for m in agent.persisted_messages]
    assert "[System: run tests]" not in persisted_contents
    assert report in persisted_contents
