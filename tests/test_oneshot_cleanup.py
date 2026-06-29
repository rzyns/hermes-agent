from __future__ import annotations

import sys
import types

from hermes_cli import oneshot


def _install_oneshot_import_fakes(monkeypatch, fake_agent_cls):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(load_config=lambda: {"model": {"model": "fake-model", "provider": "fake"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        types.SimpleNamespace(detect_provider_for_model=lambda model, provider: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda **kwargs: {
                "api_key": "fake-key",
                "base_url": "http://example.invalid/v1",
                "provider": "fake",
                "api_mode": "chat_completions",
                "credential_pool": None,
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        types.SimpleNamespace(_get_platform_tools=lambda cfg, platform: set()),
    )
    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=fake_agent_cls))
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: object())
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda cfg: [])


def test_run_agent_shuts_down_memory_provider_after_success(monkeypatch):
    instances = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self._session_messages = [{"role": "user", "content": "hello"}]
            self.shutdown_calls = []
            self.close_calls = 0
            instances.append(self)

        def run_conversation(self, prompt):
            return {"final_response": "OK"}

        def shutdown_memory_provider(self, messages=None):
            self.shutdown_calls.append(messages)

        def close(self):
            self.close_calls += 1

    _install_oneshot_import_fakes(monkeypatch, FakeAgent)

    assert oneshot._run_agent("hello", model="fake-model", provider="fake", use_config_toolsets=False) == ("OK", {"final_response": "OK"})

    assert len(instances) == 1
    assert instances[0].shutdown_calls == [[{"role": "user", "content": "hello"}]]
    assert instances[0].close_calls == 1


def test_run_agent_shuts_down_global_cached_clients(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            instances.append(self)

        def run_conversation(self, prompt):
            return {"final_response": "OK"}

    instances = []
    _install_oneshot_import_fakes(monkeypatch, FakeAgent)
    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        types.SimpleNamespace(shutdown_cached_clients=lambda: calls.append("aux")),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(shutdown_mcp_servers=lambda: calls.append("mcp")),
    )

    assert oneshot._run_agent("hello", model="fake-model", provider="fake", use_config_toolsets=False) == ("OK", {"final_response": "OK"})

    assert calls == ["mcp", "aux"]



def test_run_agent_shuts_down_memory_provider_after_chat_error(monkeypatch):
    instances = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self._session_messages = [{"role": "user", "content": "hello"}]
            self.shutdown_calls = []
            self.close_calls = 0
            instances.append(self)

        def run_conversation(self, prompt):
            raise RuntimeError("boom")

        def shutdown_memory_provider(self, messages=None):
            self.shutdown_calls.append(messages)

        def close(self):
            self.close_calls += 1

    _install_oneshot_import_fakes(monkeypatch, FakeAgent)

    try:
        oneshot._run_agent("hello", model="fake-model", provider="fake", use_config_toolsets=False)
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected chat failure")

    assert len(instances) == 1
    assert instances[0].shutdown_calls == [[{"role": "user", "content": "hello"}]]
    assert instances[0].close_calls == 1
