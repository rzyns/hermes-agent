"""Tests that provider is forwarded in on_session_start/pre_llm_call/post_llm_call hooks.

These tests exercise the wiring inside ``agent.conversation_loop.run_conversation``
and verify backward-compatible payload shapes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model="gpt-5",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        object.__setattr__(a, "session_id", "sess-1")
        return a


class TestProviderInHookPayloads:
    def test_on_session_start_includes_provider(self, agent, monkeypatch):
        captured = []

        def _capture(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)

        from agent.conversation_loop import run_conversation

        # Seed a minimal conversation_history so the loop exits immediately
        # with a no-tool-call response.
        with patch.object(
            agent.client.chat.completions, "create", return_value=_mock_response("ok")
        ):
            run_conversation(agent, "hi", conversation_history=[])

        start_calls = [kw for name, kw in captured if name == "on_session_start"]
        assert len(start_calls) == 1
        assert start_calls[0]["session_id"] == "sess-1"
        assert start_calls[0]["model"] == "gpt-5"
        assert start_calls[0]["provider"] == "openrouter"

    def test_pre_llm_call_includes_provider(self, agent, monkeypatch):
        captured = []

        def _capture(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)

        from agent.conversation_loop import run_conversation

        with patch.object(
            agent.client.chat.completions, "create", return_value=_mock_response("ok")
        ):
            run_conversation(agent, "hi", conversation_history=[])

        pre_calls = [kw for name, kw in captured if name == "pre_llm_call"]
        assert len(pre_calls) == 1
        assert pre_calls[0]["session_id"] == "sess-1"
        assert pre_calls[0]["model"] == "gpt-5"
        assert pre_calls[0]["provider"] == "openrouter"

    def test_post_llm_call_includes_provider(self, agent, monkeypatch):
        captured = []

        def _capture(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)

        from agent.conversation_loop import run_conversation

        with patch.object(
            agent.client.chat.completions, "create", return_value=_mock_response("ok")
        ):
            run_conversation(agent, "hi", conversation_history=[])

        post_calls = [kw for name, kw in captured if name == "post_llm_call"]
        assert len(post_calls) == 1
        assert post_calls[0]["session_id"] == "sess-1"
        assert post_calls[0]["model"] == "gpt-5"
        assert post_calls[0]["provider"] == "openrouter"
        assert post_calls[0]["assistant_response"] == "ok"

    def test_provider_none_or_empty_is_forwarded(self, agent, monkeypatch):
        """Backward compat: plugins that already accept **kwargs must not break."""
        captured = []

        def _capture(hook_name, **kwargs):
            captured.append((hook_name, kwargs))
            return []

        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture)
        agent.provider = None

        from agent.conversation_loop import run_conversation

        with patch.object(
            agent.client.chat.completions, "create", return_value=_mock_response("ok")
        ):
            run_conversation(agent, "hi", conversation_history=[])

        start_calls = [kw for name, kw in captured if name == "on_session_start"]
        assert start_calls[0].get("provider") is None

        pre_calls = [kw for name, kw in captured if name == "pre_llm_call"]
        assert pre_calls[0].get("provider") is None

        post_calls = [kw for name, kw in captured if name == "post_llm_call"]
        assert post_calls[0].get("provider") is None


def _mock_response(content: str):
    """Build a minimal chat completion response object."""
    from types import SimpleNamespace
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=[], reasoning=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="test/model")
    resp.usage = None
    return resp
