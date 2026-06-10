"""PluginManager smoke test for the kanban-cross-deps plugin.

Mirrors the discovery pattern used by bundled plugins (security-guidance,
nemo-relay, etc.): write a minimal config.yaml, force re-discovery, and
verify the plugin is loaded, its hooks registered, and its dependency provider
exported.

Uses a temp HERMES_HOME so no live state is touched.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import PluginManager


def _clear_kanban_cross_deps_discovery_cache(monkeypatch):
    """Reset plugin discovery without replacing ``hermes_cli.plugins``.

    Other test modules import functions from ``hermes_cli.plugins`` at module
    import time. Replacing the core module object leaves those functions bound
    to old globals while later string-based monkeypatches target the new module.
    Reset the singleton in-place and evict only plugin implementation modules.
    """

    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_plugin_manager", None)
    for key in list(sys.modules):
        if (
            key.startswith("hermes_plugins.")
            or key == "plugins.kanban_cross_deps"
            or key.startswith("plugins.kanban_cross_deps.")
        ):
            del sys.modules[key]


@pytest.fixture
def _isolate_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME with minimal config.yaml scaffold."""
    env_home = tmp_path / ".hermes"
    env_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    monkeypatch.setattr(Path, "home", lambda: env_home)
    # Prevent leaked kanban overrides from affecting board/registry resolution
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    return env_home


class TestPluginManagerLoads:
    def test_loads_via_plugin_manager(self, _isolate_env, monkeypatch):
        """End-to-end: enable in config.yaml and verify the PluginManager
        picks it up via the standard discovery path.
        """
        config = {"plugins": {"enabled": ["kanban-cross-deps"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        _clear_kanban_cross_deps_discovery_cache(monkeypatch)

        from hermes_cli.plugins import _ensure_plugins_discovered

        mgr = _ensure_plugins_discovered(force=True)
        loaded = set()
        if hasattr(mgr, "_plugins"):
            loaded = set(mgr._plugins.keys())
        assert "kanban-cross-deps" in loaded

    def test_dependency_provider_exports(self, _isolate_env, monkeypatch):
        """When loaded, the plugin exports a dependency-provider singleton
        registered with kanban_dependencies so the dispatcher and dashboard
        can query cross-board blockers.
        """
        config = {"plugins": {"enabled": ["kanban-cross-deps"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        _clear_kanban_cross_deps_discovery_cache(monkeypatch)

        from hermes_cli import kanban_dependencies as kd
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered(force=True)
        # The plugin registers its provider on import via the
        # on_plugins_discovered hook.
        providers = list(kd._providers)
        names = {kd._provider_name(p) for p in providers}
        assert "kanban_cross_deps" in names

    def test_registry_read_only_no_db_creation(self, _isolate_env, monkeypatch):
        """A fresh temp home with no prior registry DB: read-only ops must
        not create the DB file.
        """
        config = {"plugins": {"enabled": ["kanban-cross-deps"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        _clear_kanban_cross_deps_discovery_cache(monkeypatch)

        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered(force=True)

        from plugins.kanban_cross_deps.store import CrossBoardRegistry
        reg = CrossBoardRegistry()
        # Read-only queries
        assert reg.get("nonexistent") is None
        assert reg.list_edges() == []
        assert reg.count() == 0
        assert reg.schema_version() == 0
        # DB must NOT have been created on reads
        assert not reg.path.exists()

    def test_add_creates_db_and_schema(self, _isolate_env, monkeypatch):
        """Write ops create the DB and schema; subsequent reads succeed."""
        config = {"plugins": {"enabled": ["kanban-cross-deps"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        _clear_kanban_cross_deps_discovery_cache(monkeypatch)

        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered(force=True)

        from plugins.kanban_cross_deps.store import CrossBoardRegistry
        reg = CrossBoardRegistry()
        edge = reg.add(
            parent_board="a", parent_id="p1",
            child_board="b", child_id="c1",
            kind="blocks",
        )
        assert reg.path.exists()
        assert reg.schema_version() == 1
        fetched = reg.get(edge.id)
        assert fetched is not None
        assert fetched.parent_board == "a"
