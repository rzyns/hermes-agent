from types import SimpleNamespace


from agent.turn_finalizer import notify_session_end_once


def test_notify_session_end_once_deduplicates_same_turn(monkeypatch):
    import hermes_cli.lifecycle as lifecycle_mod

    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return []

    monkeypatch.setattr(lifecycle_mod, "invoke_hook", fake_invoke_hook)
    agent = SimpleNamespace(session_id="sess-1", model="model-1", platform="kanban")

    assert notify_session_end_once(
        agent,
        effective_task_id="task-1",
        turn_id="turn-1",
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="error_response",
    ) is True
    assert notify_session_end_once(
        agent,
        effective_task_id="task-1",
        turn_id="turn-1",
        completed=False,
        interrupted=True,
    ) is False

    assert len(calls) == 1
    assert calls[0][0] == "on_session_end"
    assert calls[0][1] == {
        "session_id": "sess-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "completed": False,
        "failed": True,
        "interrupted": False,
        "turn_exit_reason": "error_response",
        "model": "model-1",
        "platform": "kanban",
    }


def test_notify_session_end_once_allows_new_turn(monkeypatch):
    import hermes_cli.lifecycle as lifecycle_mod

    calls = []
    monkeypatch.setattr(
        lifecycle_mod,
        "invoke_hook",
        lambda name, **kwargs: calls.append((name, kwargs)),
    )
    agent = SimpleNamespace(session_id="sess-1", model="model-1", platform="kanban")

    assert notify_session_end_once(
        agent,
        effective_task_id="task-1",
        turn_id="turn-1",
        completed=True,
        interrupted=False,
    ) is True
    assert notify_session_end_once(
        agent,
        effective_task_id="task-1",
        turn_id="turn-2",
        completed=True,
        interrupted=False,
    ) is True

    assert [call[1]["turn_id"] for call in calls] == ["turn-1", "turn-2"]
