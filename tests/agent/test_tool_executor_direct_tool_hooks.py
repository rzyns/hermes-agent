import json
from types import SimpleNamespace

from agent.tool_executor import execute_tool_calls_sequential
from tools.todo_tool import TodoStore


class _AllowGuardrails:
    def before_call(self, _name, _args):
        return SimpleNamespace(allows_execution=True)


class _NoCheckpoint:
    enabled = False


class _NoSubdirHints:
    def check_tool_call(self, _name, _args):
        return ""


class _FakeAgent:
    def __init__(self):
        self._interrupt_requested = False
        self.log_prefix = ""
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix_chars = 20
        self.tool_delay = 0
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._tool_guardrails = _AllowGuardrails()
        self._checkpoint_mgr = _NoCheckpoint()
        self._todo_store = TodoStore()
        self._memory_manager = None
        self._context_engine_tool_names = set()
        self._current_turn_id = "turn-1"
        self.session_id = "session-1"
        self.valid_tool_names = {"todo"}
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._subdirectory_hints = _NoSubdirHints()
        self.tool_emoji_overrides = {}
        self._current_tool = None

    def _touch_activity(self, _message):
        pass

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _vprint(self, *_args, **_kwargs):
        pass

    def _append_guardrail_observation(self, _name, _args, result, *, failed=False):
        return result

    def _record_file_mutation_result(self, *_args, **_kwargs):
        pass

    def _tool_result_content_for_active_model(self, _name, result):
        return result

    def _apply_pending_steer_to_tool_results(self, _messages, _count):
        pass

    def _flush_messages_to_session_db(self, _messages):
        pass


def _tool_call(name, args, tool_call_id="call-direct-1"):
    return SimpleNamespace(
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_sequential_direct_tool_hooks_receive_matching_ids(monkeypatch):
    """Direct sequential tools must emit matching pre/post hook identifiers.

    Regression coverage for Langfuse null tool-output spans: the sequential
    direct-tool path used to fire pre_tool_call without session/tool ids and did
    not emit post_tool_call for built-in direct tools such as todo/memory.
    """

    pre_calls = []
    hook_calls = []

    def fake_resolve_pre_tool_block(function_name, function_args, **kwargs):
        pre_calls.append((function_name, function_args, kwargs))
        return None

    def fake_invoke_hook(hook_name, **kwargs):
        hook_calls.append((hook_name, kwargs))
        return []

    monkeypatch.setattr(
        "hermes_cli.plugins.resolve_pre_tool_block",
        fake_resolve_pre_tool_block,
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    agent = _FakeAgent()
    messages = []
    assistant_message = SimpleNamespace(
        tool_calls=[
            _tool_call(
                "todo",
                {"todos": [{"id": "a", "content": "Check hooks", "status": "pending"}]},
                tool_call_id="tc-direct-todo",
            )
        ]
    )

    execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

    assert len(pre_calls) == 1
    _name, _args, pre_kwargs = pre_calls[0]
    assert pre_kwargs["task_id"] == "task-1"
    assert pre_kwargs["session_id"] == "session-1"
    assert pre_kwargs["tool_call_id"] == "tc-direct-todo"
    assert pre_kwargs["turn_id"] == "turn-1"

    post = [(name, kwargs) for name, kwargs in hook_calls if name == "post_tool_call"]
    transform = [(name, kwargs) for name, kwargs in hook_calls if name == "transform_tool_result"]
    assert len(post) == 1
    assert len(transform) == 1
    for _hook_name, kwargs in post + transform:
        assert kwargs["tool_name"] == "todo"
        assert kwargs["task_id"] == "task-1"
        assert kwargs["session_id"] == "session-1"
        assert kwargs["tool_call_id"] == "tc-direct-todo"
        assert kwargs["turn_id"] == "turn-1"
        assert isinstance(kwargs["duration_ms"], int)
        assert kwargs["duration_ms"] >= 0
        assert '"id": "a"' in kwargs["result"]

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "tc-direct-todo"


def test_sequential_direct_tool_transform_can_replace_result(monkeypatch):
    pre_calls = []

    def fake_resolve_pre_tool_block(function_name, function_args, **kwargs):
        pre_calls.append((function_name, function_args, kwargs))
        return None

    def fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_tool_result":
            return ['{"transformed": true}']
        return []

    monkeypatch.setattr(
        "hermes_cli.plugins.resolve_pre_tool_block",
        fake_resolve_pre_tool_block,
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    agent = _FakeAgent()
    messages = []
    assistant_message = SimpleNamespace(
        tool_calls=[_tool_call("todo", {"todos": []}, tool_call_id="tc-transform")]
    )

    execute_tool_calls_sequential(agent, assistant_message, messages, "task-2")

    assert messages[0]["content"] == '{"transformed": true}'
