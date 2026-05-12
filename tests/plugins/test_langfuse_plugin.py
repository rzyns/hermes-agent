"""Tests for the bundled observability/langfuse plugin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "langfuse"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "langfuse"
        assert data["version"]
        # All six hooks the plugin implements.
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        }
        # Required env vars are the user-facing HERMES_ prefixed keys.
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in data["requires_env"]
        assert "HERMES_LANGFUSE_SECRET_KEY" in data["requires_env"]


# ---------------------------------------------------------------------------
# Plugin discovery: langfuse is opt-in (not loaded unless explicitly enabled).
# This guards against someone accidentally re-introducing a per-hook
# load_config() gate or making the plugin auto-load.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # observability/langfuse appears in the plugin registry …
        loaded = manager._plugins.get("observability/langfuse")
        assert loaded is not None, "plugin not discovered"
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Runtime gate: _get_langfuse() returns None and caches _INIT_FAILED when
# credentials are missing. Guards against regressing toward the rejected
# per-hook load_config() design.
# ---------------------------------------------------------------------------

class TestRuntimeGate:
    def _fresh_plugin(self):
        """Import the plugin module fresh (clears any cached client)."""
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_get_langfuse_returns_none_without_credentials(self, monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()
        assert langfuse_plugin._get_langfuse() is None

    def test_get_langfuse_caches_failure_no_config_load(self, monkeypatch):
        """A miss must be cached — no per-hook config.yaml reads, no env re-reads."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()

        # Prime the cache with one call.
        assert langfuse_plugin._get_langfuse() is None

        # Now block os.environ.get — a correctly-cached plugin must not
        # touch env again.
        import os
        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_LANGFUSE_", "LANGFUSE_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)

        for _ in range(20):
            assert langfuse_plugin._get_langfuse() is None

        assert called["n"] == 0, (
            f"_get_langfuse() re-read env {called['n']} times after cache miss — "
            "it should short-circuit via _INIT_FAILED"
        )

    def test_get_langfuse_does_not_import_hermes_config(self, monkeypatch):
        """The plugin must not re-read config.yaml per hook."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        # Drop any cached import of hermes_cli.config.
        sys.modules.pop("hermes_cli.config", None)

        langfuse_plugin = self._fresh_plugin()
        for _ in range(20):
            langfuse_plugin._get_langfuse()

        assert "hermes_cli.config" not in sys.modules, (
            "langfuse plugin imported hermes_cli.config — regression toward "
            "the rejected per-hook load_config() design"
        )


# ---------------------------------------------------------------------------
# Hooks are inert when the client is unavailable.
# ---------------------------------------------------------------------------

class TestHooksInert:
    def test_hooks_noop_without_client(self, monkeypatch):
        """All 6 hooks must return without raising when _get_langfuse() is None."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        # Each hook should just return; no exceptions.
        mod.on_pre_llm_call(task_id="t", session_id="s", messages=[{"role": "user", "content": "hi"}])
        mod.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        mod.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")


# ---------------------------------------------------------------------------
# LF11 turn-scoped trace metadata + tool correlation
# ---------------------------------------------------------------------------

class _FakeObservation:
    def __init__(self, observation_name="root", **kwargs):
        self.name = observation_name
        self.kwargs = kwargs
        self.children = []
        self.updates = []
        self.ended = False
        self.trace_io = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def start_observation(self, **kwargs):
        obs = _FakeObservation(kwargs.get("name", "child"), **kwargs)
        self.children.append(obs)
        return obs

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True

    def set_trace_io(self, **kwargs):
        self.trace_io.update(kwargs)


class _FakeLangfuse:
    def __init__(self):
        self.roots = []
        self.flushed = False

    def create_trace_id(self, seed):
        return "trace-" + seed.replace(":", "-")[:80]

    def start_as_current_observation(self, **kwargs):
        obs = _FakeObservation(kwargs.get("name", "root"), **kwargs)
        self.roots.append(obs)
        return obs

    def flush(self):
        self.flushed = True


class TestLF11TurnTraceContract:
    def _fresh_plugin(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")
        fake = _FakeLangfuse()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: fake)
        monkeypatch.setattr(mod, "propagate_attributes", None)
        mod._TRACE_STATE.clear()
        return mod, fake

    def test_trace_key_prefers_turn_id_and_preserves_legacy_fallbacks(self):
        mod = importlib.import_module("plugins.observability.langfuse")
        assert mod._trace_key(turn_id="turn_a", task_id="task_a", session_id="session_a") == "turn:turn_a"
        assert mod._trace_key(task_id="task_a", session_id="session_a") == "task:task_a"
        assert mod._trace_key(session_id="session_a") == "session:session_a"

    def test_two_turns_share_session_but_create_distinct_trace_states(self, monkeypatch):
        mod, fake = self._fresh_plugin(monkeypatch)
        for turn in ("turn_one", "turn_two"):
            mod.on_pre_llm_request(
                task_id="task_same",
                session_id="session_same",
                turn_id=turn,
                platform="cli",
                surface="cli",
                profile="backend-eng",
                provider="openai",
                model="gpt-test",
                api_mode="chat_completions",
                api_call_count=1,
                messages=[{"role": "user", "content": "hello"}],
            )
        assert set(mod._TRACE_STATE) == {"turn:turn_one", "turn:turn_two"}
        assert len(fake.roots) == 2
        for root, turn in zip(fake.roots, ("turn_one", "turn_two")):
            assert root.kwargs["trace_context"]["session_id"] == "session_same"
            assert root.kwargs["name"] == "Hermes cli turn"
            metadata = root.kwargs["metadata"]
            assert metadata["metadata_schema_version"] == "hermes.langfuse.v1"
            assert metadata["turn_id"] == turn
            assert metadata["session_id"] == "session_same"
            assert metadata["task_id"] == "task_same"
            assert metadata["profile"] == "backend-eng"
            assert metadata["surface"] == "cli"
            assert metadata["provider"] == "openai"
            assert metadata["model"] == "gpt-test"
            assert metadata["api_mode"] == "chat_completions"

    def test_tool_exact_fallback_duplicate_and_ambiguous_correlation(self, monkeypatch):
        mod, fake = self._fresh_plugin(monkeypatch)
        mod.on_pre_llm_request(
            task_id="task", session_id="session", turn_id="turn", platform="cli",
            api_call_count=1, messages=[{"role": "user", "content": "hi"}],
        )
        # Exact id match closes with output.
        mod.on_pre_tool_call(tool_name="terminal", args={"command": "date"}, task_id="task", session_id="session", turn_id="turn", tool_call_id="tc1")
        mod.on_post_tool_call(tool_name="terminal", args={"command": "date"}, result='{"output":"ok"}', task_id="task", session_id="session", turn_id="turn", tool_call_id="tc1")
        assert any(child.updates and child.updates[-1]["output"]["output"] == "ok" for child in fake.roots[0].children)
        assert "tc1" not in mod._TRACE_STATE["turn:turn"].tools

        # Duplicate pre with same real id is idempotent: only one open tool span.
        mod.on_pre_tool_call(tool_name="read_file", args={"path": "a"}, task_id="task", session_id="session", turn_id="turn", tool_call_id="tc2")
        mod.on_pre_tool_call(tool_name="read_file", args={"path": "a"}, task_id="task", session_id="session", turn_id="turn", tool_call_id="tc2")
        assert list(mod._TRACE_STATE["turn:turn"].tools) == ["tc2"]
        mod.on_post_tool_call(tool_name="read_file", args={"path": "a"}, result="done", task_id="task", session_id="session", turn_id="turn", tool_call_id="tc2")
        assert "tc2" not in mod._TRACE_STATE["turn:turn"].tools

        # Fallback-keyed spans disambiguate by args fingerprint.
        mod.on_pre_tool_call(tool_name="terminal", args={"command": "one"}, task_id="task", session_id="session", turn_id="turn")
        mod.on_pre_tool_call(tool_name="terminal", args={"command": "two"}, task_id="task", session_id="session", turn_id="turn")
        mod.on_post_tool_call(tool_name="terminal", args={"command": "two"}, result="two-result", task_id="task", session_id="session", turn_id="turn", tool_call_id="missing")
        remaining = list(mod._TRACE_STATE["turn:turn"].tools.values())
        assert len(remaining) == 1
        assert remaining[0].args_fingerprint == mod._args_fingerprint({"command": "one"})

        # Ambiguous same-name/same-args fallback remains unassigned.
        mod.on_pre_tool_call(tool_name="browser", args={"url": "x"}, task_id="task", session_id="session", turn_id="turn")
        mod.on_pre_tool_call(tool_name="browser", args={"url": "x"}, task_id="task", session_id="session", turn_id="turn")
        before = len(mod._TRACE_STATE["turn:turn"].tools)
        mod.on_post_tool_call(tool_name="browser", args={"url": "x"}, result="ambiguous", task_id="task", session_id="session", turn_id="turn", tool_call_id="missing")
        assert len(mod._TRACE_STATE["turn:turn"].tools) == before
