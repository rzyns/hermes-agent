"""Startup diagnostics must not mistake pending plugin discovery for bad config."""

import json
import threading
from unittest.mock import Mock

import pytest


def test_pending_plugin_names_do_not_warn_or_block_startup(tmp_path, monkeypatch, caplog):
    import cli
    from hermes_cli import plugins, tools_config
    from toolsets import validate_toolset

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    cache = tmp_path / "cache" / "plugin_toolset_keys.json"
    cache.parent.mkdir()
    pending = ["pending_native_plugin", "agent-plugin-pending__worker"]
    cache.write_text(json.dumps({"toolset_keys": pending[:1], "portable_mcp": pending[1:]}))
    assert not any(validate_toolset(name) for name in pending)

    # Hold discovery in flight with an Event, not a timing-dependent sleep.
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    monkeypatch.setattr(plugins, "_background_discovery_thread", thread)
    discover = Mock(side_effect=AssertionError("startup must not join warm discovery"))
    monkeypatch.setattr(plugins, "discover_plugins", discover)
    monkeypatch.setattr(tools_config, "_warned_invalid_platform_toolsets", set())
    monkeypatch.setattr(cli, "CLI_CONFIG", {
        "agent": {"disabled_toolsets": ["terminal"]},
        "mcp_servers": {"configured_mcp": {"enabled": False}},
    })
    instance = cli.HermesCLI.__new__(cli.HermesCLI)
    instance._console_print = Mock()
    selected = ["terminal", *pending, "configured_mcp", "toolset_typo"]
    thread.start()
    try:
        instance._init_toolsets(selected)
        instance._console_print.assert_called_once_with(
            "[bold red]Warning: Unknown toolsets: toolset_typo[/]"
        )
        assert instance.enabled_toolsets == selected
        assert instance.disabled_toolsets == ["terminal"]
        for name in pending:
            config = {"platform_toolsets": {"cli": [name]}}
            assert name in tools_config._get_platform_tools(config, "cli")
        assert "no valid toolsets configured" not in caplog.text
        assert not any(validate_toolset(name) for name in pending)

        for selection in (None, [], ["all"], ["*"], ["terminal"]):
            instance._console_print.reset_mock()
            instance._init_toolsets(selection)
            instance._console_print.assert_not_called()
            assert instance.enabled_toolsets == selection
        discover.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.parametrize("cache_contents", [None, "not json", '{"toolset_keys": ["removed_plugin"], "portable_mcp": ["removed_mcp"]}'])
def test_discovered_plugins_and_mcp_names_are_known_but_stale_cache_is_not(
    tmp_path, monkeypatch, caplog, cache_contents,
):
    import cli
    from hermes_cli import plugins, tools_config
    from hermes_cli.agent_plugins import MCP_SCHEMA_V1, PLUGIN_SCHEMA_V1

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    manager = plugins.PluginManager()
    monkeypatch.setattr(manager, "_scan_entry_points", lambda: [])
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "_background_discovery_thread", None)

    native = tmp_path / "plugins" / "startup-native"
    native.mkdir(parents=True)
    (native / "plugin.yaml").write_text("name: startup-native\nversion: 1.0.0\n")
    (native / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(name='startup_probe', toolset='startup_native',\n"
        "        schema={'name': 'startup_probe', 'description': 'Test probe',\n"
        "                'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kwargs: {'ok': True})\n"
    )
    portable = tmp_path / "plugins" / "startup-portable"
    portable.mkdir()
    (portable / "plugin.json").write_text(json.dumps({
        "$schema": PLUGIN_SCHEMA_V1, "name": "startup.portable",
    }))
    (portable / "mcp.json").write_text(json.dumps({
        "$schema": MCP_SCHEMA_V1,
        "mcpServers": {"worker": {"type": "stdio", "command": "unused-test-server"}},
    }))
    config = {
        "agent": {},
        "plugins": {"enabled": ["startup-native", "startup.portable"]},
        "mcp_servers": {"configured_mcp": {"command": "unused-test-server"}},
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))
    if cache_contents is not None:
        cache = tmp_path / "cache" / "plugin_toolset_keys.json"
        cache.parent.mkdir()
        cache.write_text(cache_contents)
    monkeypatch.setattr(cli, "CLI_CONFIG", config)
    monkeypatch.setattr(tools_config, "_warned_invalid_platform_toolsets", set())
    instance = cli.HermesCLI.__new__(cli.HermesCLI)
    instance._console_print = Mock()
    try:
        # No cache or live discovery: real imports must discover the native plugin.
        instance._init_toolsets(["startup_native"])
        instance._console_print.assert_not_called()
        assert manager._discovered
        portable_names = set(manager.get_portable_mcp_servers())
        assert portable_names
        assert not any(p["error"] for p in manager.list_plugins())

        selected = ["startup_native", *sorted(portable_names), "configured_mcp"]
        instance._init_toolsets(selected + ["removed_plugin", "removed_mcp"])
        instance._console_print.assert_called_once_with(
            "[bold red]Warning: Unknown toolsets: removed_plugin, removed_mcp[/]"
        )
        assert instance.enabled_toolsets == selected + ["removed_plugin", "removed_mcp"]

        # A plugin-only or MCP-only platform is not an all-invalid selection.
        for name in selected:
            config["platform_toolsets"] = {"cli": [name]}
            assert name in tools_config._get_platform_tools(config, "cli")
        assert "no valid toolsets configured" not in caplog.text

        config["platform_toolsets"] = {"cli": ["toolset_typo"]}
        tools_config._get_platform_tools(config, "cli")
        assert "no valid toolsets configured" in caplog.text
        assert "toolset_typo" in caplog.text
    finally:
        manager.unload()
