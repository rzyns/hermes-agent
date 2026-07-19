"""Startup-level tests for ``hermes proxy`` plugin dependencies."""

from __future__ import annotations

from types import SimpleNamespace

from agent import image_gen_registry
from hermes_cli import plugins as plugins_module
from hermes_cli.plugins import PluginManager
from hermes_cli.proxy import cli as proxy_cli


def test_proxy_start_discovers_real_codex_image_provider(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - image_gen/openai-codex\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(plugins_module, "_plugin_manager", PluginManager())
    image_gen_registry._reset_for_tests()

    class AuthenticatedAdapter:
        display_name = "Test Subscription"
        name = "subscription"

        def is_authenticated(self):
            return True

    monkeypatch.setattr(
        proxy_cli,
        "get_adapter",
        lambda provider: AuthenticatedAdapter(),
    )

    async def assert_provider_registered(adapter, host, port):
        provider = image_gen_registry.get_provider("openai-codex")
        assert provider is not None
        assert provider.name == "openai-codex"

    monkeypatch.setattr(proxy_cli, "run_server", assert_provider_registered)

    try:
        result = proxy_cli.cmd_proxy_start(
            SimpleNamespace(
                provider="subscription",
                host="127.0.0.1",
                port=0,
            )
        )
    finally:
        image_gen_registry._reset_for_tests()

    assert result == 0


def test_proxy_start_fails_clearly_when_plugin_discovery_crashes(
    monkeypatch,
    capsys,
):
    class AuthenticatedAdapter:
        display_name = "Test Subscription"
        name = "subscription"

        def is_authenticated(self):
            return True

    monkeypatch.setattr(
        proxy_cli,
        "get_adapter",
        lambda provider: AuthenticatedAdapter(),
    )

    def fail_discovery():
        raise RuntimeError("discovery unavailable")

    monkeypatch.setattr(plugins_module, "discover_plugins", fail_discovery)

    async def server_must_not_start(*args, **kwargs):
        raise AssertionError("server must not start without plugin discovery")

    monkeypatch.setattr(proxy_cli, "run_server", server_must_not_start)

    result = proxy_cli.cmd_proxy_start(
        SimpleNamespace(
            provider="subscription",
            host="127.0.0.1",
            port=8645,
        )
    )

    assert result == 1
    assert "proxy: plugin discovery failed: discovery unavailable" in capsys.readouterr().err
