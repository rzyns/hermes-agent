from __future__ import annotations

from types import SimpleNamespace

from agent.conversation_loop import (
    _ollama_context_limit_error_for_tools,
    _select_tool_schemas_for_request,
)
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
from hermes_cli import plugins
from hermes_cli.plugins import PluginManager, VALID_HOOKS


def test_select_tool_schemas_is_valid_hook():
    assert "select_tool_schemas" in VALID_HOOKS


def test_plugin_manager_invoke_hook_filters_none_and_keeps_first_list():
    mgr = PluginManager()
    original = [{"type": "function", "function": {"name": "a"}}]
    selected = [{"type": "function", "function": {"name": "b"}}]
    mgr._hooks["select_tool_schemas"] = [lambda **kwargs: None, lambda **kwargs: selected]

    results = mgr.invoke_hook("select_tool_schemas", schemas=original)

    assert results[0] == selected


def test_plugin_manager_invoke_hook_exception_fails_open_to_other_results():
    mgr = PluginManager()
    selected = [{"type": "function", "function": {"name": "safe"}}]

    def boom(**kwargs):
        raise RuntimeError("broken selector")

    mgr._hooks["select_tool_schemas"] = [boom, lambda **kwargs: selected]

    assert mgr.invoke_hook("select_tool_schemas", schemas=[]) == [selected]


def test_select_tool_schemas_uses_first_list_and_passes_request_copy(monkeypatch):
    original = [{"name": "read_file"}, {"name": "search_files"}]
    selected = [{"name": "read_file"}]
    agent = SimpleNamespace(
        tools=original,
        session_id="session-1",
        model="model-1",
        platform="cli",
        provider="openai-codex",
    )
    seen = {}

    def fake_invoke(hook_name, **kwargs):
        seen.update(kwargs)
        assert hook_name == "select_tool_schemas"
        return ["ignore-me", selected, [{"name": "second-selector"}]]

    monkeypatch.setattr(plugins, "invoke_hook", fake_invoke)

    result = _select_tool_schemas_for_request(
        agent,
        "inspect files",
        [{"role": "user", "content": "inspect files"}],
    )

    assert result == selected
    assert seen["schemas"] == original
    assert seen["schemas"] is not original
    assert agent.tools is original


def test_select_tool_schemas_empty_list_is_valid_selection(monkeypatch):
    agent = SimpleNamespace(
        tools=[{"name": "read_file"}],
        session_id="session-1",
        model="model-1",
        platform="cli",
        provider="openai-codex",
    )
    monkeypatch.setattr(plugins, "invoke_hook", lambda hook_name, **kwargs: [[]])

    assert _select_tool_schemas_for_request(agent, "no tools", []) == []
    assert agent.tools == [{"name": "read_file"}]


def test_select_tool_schemas_no_list_result_fails_open_to_copy(monkeypatch):
    original = [{"name": "read_file"}]
    agent = SimpleNamespace(
        tools=original,
        session_id="session-1",
        model="model-1",
        platform="cli",
        provider="openai-codex",
    )
    monkeypatch.setattr(plugins, "invoke_hook", lambda hook_name, **kwargs: ["not-a-list"])

    result = _select_tool_schemas_for_request(agent, "read", [])

    assert result == original
    assert result is not original
    assert agent.tools is original


def test_select_tool_schemas_hook_invoke_error_fails_open_to_copy(monkeypatch):
    original = [{"name": "read_file"}]
    agent = SimpleNamespace(
        tools=original,
        session_id="session-1",
        model="model-1",
        platform="cli",
        provider="openai-codex",
    )

    def boom(*args, **kwargs):
        raise RuntimeError("manager unavailable")

    monkeypatch.setattr(plugins, "invoke_hook", boom)

    result = _select_tool_schemas_for_request(agent, "read", [])

    assert result == original
    assert result is not original
    assert agent.tools is original


def test_request_local_empty_selection_disables_ollama_tool_context_error():
    agent = SimpleNamespace(
        tools=[{"name": "read_file"}],
        _ollama_num_ctx=MINIMUM_CONTEXT_LENGTH - 1,
        model="ollama-model",
        base_url="http://localhost:11434",
        provider="ollama",
        session_id="session-1",
    )

    assert _ollama_context_limit_error_for_tools(agent, 100_000, []) is None
    assert _ollama_context_limit_error_for_tools(agent, 100_000, agent.tools)
