"""Tests for the bundled observability/langfuse plugin."""
from __future__ import annotations

import importlib
import logging
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
        # Request, tool, and session lifecycle hooks the plugin implements.
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
            "on_session_end",
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
        mod.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, request_messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        mod.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")


# ---------------------------------------------------------------------------
# Placeholder-credential guard (#23823).
#
# Regression coverage for the silent-failure bug: when an operator leaves
# HERMES_LANGFUSE_PUBLIC_KEY / SECRET_KEY at a template value like
# "placeholder", "test-key", or "your-langfuse-key", the SDK accepts the
# credentials at construction time (it does no server-side validation
# eagerly) but drops every trace at flush time, with no signal in the
# Hermes logs.  The fix in `_get_langfuse()` validates the documented
# `pk-lf-` / `sk-lf-` prefix Langfuse always issues, surfaces a one-shot
# warning naming the offending env var(s), and short-circuits via the
# same `_INIT_FAILED` path used for missing credentials so subsequent
# hook invocations don't re-log.
# ---------------------------------------------------------------------------


class _FakeLangfuse:
    """Stand-in for the real :class:`langfuse.Langfuse` so tests don't
    need the optional ``langfuse`` SDK installed.  The plugin's runtime
    gate refuses to proceed past ``if Langfuse is None`` when the SDK
    is missing, which would short-circuit before the placeholder check
    can fire.  Patching ``plugin.Langfuse`` with this class lets the
    placeholder validator exercise its full code path."""

    instances: list["_FakeLangfuse"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeLangfuse.instances.append(self)


class TestPlaceholderKeyDetection:
    LOGGER_NAME = "plugins.observability.langfuse"

    def _fresh_plugin(self, monkeypatch=None):
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)
        if monkeypatch is not None:
            # Pretend the SDK is installed so `_get_langfuse()` actually
            # reaches the placeholder check.  Real SDK calls are never
            # made because the placeholder/missing-credentials paths
            # return before constructing a client.
            _FakeLangfuse.instances.clear()
            monkeypatch.setattr(mod, "Langfuse", _FakeLangfuse, raising=False)
        return mod

    @staticmethod
    def _clear_env(monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    # -- helper unit tests (no SDK stub needed: these don't go through
    #    _get_langfuse, they exercise the pure-Python helpers directly) ------

    def test_redact_key_preview_empty(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._redact_key_preview("") == "<empty>"

    def test_redact_key_preview_short_value_echoed(self, monkeypatch):
        """Short placeholder strings are echoed in full so the operator
        can see exactly which template they forgot to replace."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._redact_key_preview("placeholder") == "'placeholder'"
        assert plugin._redact_key_preview("test-key") == "'test-key'"

    def test_redact_key_preview_long_value_truncated(self, monkeypatch):
        """If an operator pasted a real secret into the wrong env var the
        preview must NOT echo it in full — only the leading 6 chars."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        result = plugin._redact_key_preview("sk-lf-abcdefghijklmnop")
        assert "abcdefghij" not in result
        assert result.startswith("'sk-lf-")
        assert result.endswith("...'")

    def test_validate_langfuse_key_accepts_documented_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz"
        ) is None
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz"
        ) is None

    def test_validate_langfuse_key_rejects_wrong_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        msg = plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_PUBLIC_KEY", "placeholder"
        )
        assert msg is not None
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in msg
        assert "pk-lf-" in msg

    def test_validate_langfuse_key_unknown_name_passes(self, monkeypatch):
        """Defensive: an env var with no registered prefix is trusted."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._validate_langfuse_key("HERMES_LANGFUSE_BASE_URL", "anything") is None

    # -- end-to-end _get_langfuse() behaviour --------------------------------
    # These tests pass `monkeypatch` to _fresh_plugin() so the helper can
    # stub out `Langfuse` (the optional SDK).  Without that, every call
    # short-circuits at `if Langfuse is None` before reaching the
    # placeholder validator — masking the very behaviour we're testing.

    def test_placeholder_public_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "'placeholder'" in text
        assert "pk-lf-" in text
        # The valid secret value must NOT appear (the var NAME does, in
        # the "or unset ..." hint, but the value preview shouldn't).
        assert "'sk-lf-" not in text
        # Never constructed the SDK client — short-circuited before that.
        assert _FakeLangfuse.instances == []

    def test_placeholder_secret_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "test-key")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text
        assert "'test-key'" in text
        assert "sk-lf-" in text
        # The valid public value must NOT appear.
        assert "'pk-lf-" not in text
        assert _FakeLangfuse.instances == []

    def test_both_placeholders_one_warning_with_both_keys(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Expected a single combined warning; got {len(warnings)}:\n"
            + "\n".join(r.getMessage() for r in warnings)
        )
        text = warnings[0].getMessage()
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text

    def test_repeated_calls_do_not_re_warn(self, monkeypatch, caplog):
        """The cached ``_INIT_FAILED`` sentinel must short-circuit
        subsequent calls so each hook invocation isn't a fresh log
        line — otherwise a busy gateway will spam the operator's
        terminal."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            for _ in range(15):
                assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Warning fired {len(warnings)} times across 15 calls; "
            "expected 1 (cached via _INIT_FAILED)"
        )

    @pytest.mark.parametrize("placeholder", [
        "placeholder",
        "test-key",
        "your-langfuse-key",
        "change-me",
        "xxx",
        "dummy-key-here",
        "<your-key>",
        "REPLACE_ME",
    ])
    def test_common_placeholders_detected(self, monkeypatch, caplog, placeholder):
        """A grab-bag of values that real-world ``.env.example`` templates
        use as stand-ins.  Any of them in either key must trip the guard."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", placeholder)
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in caplog.text

    def test_legacy_LANGFUSE_PUBLIC_KEY_also_validated(self, monkeypatch, caplog):
        """The plugin reads both the canonical HERMES_-prefixed env var and
        the legacy bare ``LANGFUSE_PUBLIC_KEY``.  The validator must run on
        whichever value ``_get_langfuse()`` actually consumed."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        # Warning names the canonical user-facing env var (the bare
        # LANGFUSE_PUBLIC_KEY is a backwards-compat alias for the
        # HERMES_-prefixed one — operators set the HERMES_-prefixed one).
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in caplog.text
        assert "'placeholder'" in caplog.text

    def test_missing_credentials_still_skip_silently(self, monkeypatch, caplog):
        """Missing-creds is the documented opt-out path (operator hasn't
        configured the plugin yet) — it must remain SILENT.  Regression
        guard against the placeholder validator accidentally running on
        empty values and re-introducing log noise for unconfigured
        installs."""
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert warnings == []

    def test_sdk_not_installed_still_skips_silently(self, monkeypatch, caplog):
        """If the langfuse SDK isn't installed at all, the placeholder
        check should never run — there's nothing the operator can do
        about a credential mismatch when the package is missing, and
        re-warning here would dilute the actually-actionable SDK-missing
        signal upstream.  The ``Langfuse is None`` guard at the top of
        ``_get_langfuse`` already handles this; this test pins that
        behaviour."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        # NO monkeypatch on Langfuse here — falls back to whatever the
        # plugin imported at module load (None if SDK absent).
        plugin = self._fresh_plugin()
        monkeypatch.setattr(plugin, "Langfuse", None, raising=False)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert warnings == []

    def test_valid_prefixes_do_not_trigger_placeholder_warning(self, monkeypatch, caplog):
        """Real Langfuse keys (``pk-lf-…`` / ``sk-lf-…``) must pass the
        guard and proceed to SDK init.  We stub the SDK constructor with
        a recording fake so the assertion can confirm BOTH that the
        placeholder warning didn't fire AND that the client was actually
        constructed — the latter is the success signal the bug report
        wanted."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            client = plugin._get_langfuse()
        assert isinstance(client, _FakeLangfuse)
        assert client.kwargs["public_key"] == "pk-lf-real-public-xyz"
        assert client.kwargs["secret_key"] == "sk-lf-real-secret-xyz"
        assert "placeholders" not in caplog.text.lower(), (
            f"Valid Langfuse keys tripped the placeholder guard: {caplog.text!r}"
        )


class TestRequestMessageCoercion:
    def test_prefers_request_messages_then_messages_then_history_then_user_message(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        assert mod._coerce_request_messages(
            request_messages=[{"role": "system", "content": "s"}],
            messages=[{"role": "user", "content": "m"}],
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "system", "content": "s"}]
        assert mod._coerce_request_messages(
            messages=[{"role": "user", "content": "m"}],
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "user", "content": "m"}]
        assert mod._coerce_request_messages(
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "user", "content": "h"}]
        assert mod._coerce_request_messages(user_message="u") == [{"role": "user", "content": "u"}]


class TestToolCallOutputBackfill:
    def test_post_tool_call_backfills_matching_turn_tool_call_output(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        observation = object()
        state = mod.TraceState(trace_id="trace-1", root_ctx=None, root_span=None)
        state.tools["call-1"] = observation
        state.turn_tool_calls.append({
            "id": "call-1",
            "type": "function",
            "name": "web_extract",
            "arguments": '{"urls": ["https://example.com"]}',
            "function": {
                "name": "web_extract",
                "arguments": '{"urls": ["https://example.com"]}',
            },
        })

        task_key = mod._trace_key("task-1", "session-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end_observation(obs, *, output=None, metadata=None, usage_details=None, cost_details=None):
            ended["observation"] = obs
            ended["output"] = output
            ended["metadata"] = metadata

        monkeypatch.setattr(mod, "_end_observation", fake_end_observation)

        mod.on_post_tool_call(
            tool_name="web_extract",
            args={"urls": ["https://example.com"]},
            result='{"results": [{"url": "https://example.com", "content": "Example Domain"}]}',
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
        )

        assert ended["observation"] is observation
        assert state.turn_tool_calls[0]["output"] == ended["output"]
        assert state.turn_tool_calls[0]["function"]["output"] == ended["output"]
        assert state.turn_tool_calls[0]["output"] == {
            "results": [{"url": "https://example.com", "content": "Example Domain"}]
        }

    def test_serialize_messages_keeps_tool_name_and_call_id(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        messages = [{
            "role": "tool",
            "name": "web_extract",
            "tool_call_id": "call-1",
            "content": '{"ok": true}',
        }]

        assert mod._serialize_messages(messages) == [{
            "role": "tool",
            "name": "web_extract",
            "tool_call_id": "call-1",
            "content": {"ok": True},
        }]

    def test_serialize_tool_calls_emits_openai_style_function_shape(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        class _Fn:
            name = "web_extract"
            arguments = '{"urls": ["https://example.com"]}'

        class _ToolCall:
            id = "call-1"
            type = "function"
            function = _Fn()

        assert mod._serialize_tool_calls([_ToolCall()]) == [{
            "id": "call-1",
            "type": "function",
            "name": "web_extract",
            "arguments": '{"urls": ["https://example.com"]}',
            "function": {
                "name": "web_extract",
                "arguments": '{"urls": ["https://example.com"]}',
            },
        }]


class TestToolObservationKeying:
    """Tests for pre/post tool_call observation matching when tool_call_id is absent."""

    def _make_mod(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    def test_empty_tool_call_id_single_tool_sets_output(self, monkeypatch):
        mod = self._make_mod()
        obs = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.pending_tools_by_name.setdefault("my_tool", []).append(obs)

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="my_tool",
            args={},
            result='{"ok": true}',
            task_id="task-1",
            session_id="sess-1",
            tool_call_id="",
        )

        assert ended["obs"] is obs
        assert ended["output"] == {"ok": True}
        assert state.pending_tools_by_name.get("my_tool") is None

    def test_empty_tool_call_id_observations_are_fifo_within_tool_name(self, monkeypatch):
        """Two queued observations are consumed in FIFO order so the first
        post hook gets the first observation's output, not the second.

        Sequential-on-one-thread coverage; the real concurrent case is
        guarded by ``_STATE_LOCK`` around every read-modify-write on
        ``pending_tools_by_name`` and is exercised in
        ``test_threaded_post_calls_preserve_fifo_under_lock`` below.
        """
        mod = self._make_mod()
        obs_a, obs_b = object(), object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.pending_tools_by_name["web_extract"] = [obs_a, obs_b]

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        calls = []

        def fake_end(o, *, output=None, metadata=None, **kw):
            calls.append((o, output))

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="web_extract", args={}, result='{"val": "a"}',
            task_id="task-1", session_id="sess-1", tool_call_id="",
        )
        mod.on_post_tool_call(
            tool_name="web_extract", args={}, result='{"val": "b"}',
            task_id="task-1", session_id="sess-1", tool_call_id="",
        )

        assert calls[0] == (obs_a, {"val": "a"})
        assert calls[1] == (obs_b, {"val": "b"})
        assert state.pending_tools_by_name.get("web_extract") is None

    def test_threaded_post_calls_preserve_fifo_under_lock(self, monkeypatch):
        """The actual concurrency contract: when 8 threads race to drain
        the pending queue, no observation is consumed twice and none is
        lost.  Validates ``_STATE_LOCK`` discipline, not Python list
        semantics."""
        import threading

        mod = self._make_mod()
        n = 8
        observations = [object() for _ in range(n)]
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.pending_tools_by_name["web_extract"] = list(observations)

        task_key = mod._trace_key("task-thr", "sess-thr")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        recorded: list = []
        lock = threading.Lock()

        def fake_end(o, *, output=None, metadata=None, **kw):
            with lock:
                recorded.append(o)

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        barrier = threading.Barrier(n)

        def worker():
            barrier.wait()
            mod.on_post_tool_call(
                tool_name="web_extract", args={}, result='{"ok": true}',
                task_id="task-thr", session_id="sess-thr", tool_call_id="",
            )

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every observation was consumed exactly once; queue is empty.
        assert len(recorded) == n
        assert set(map(id, recorded)) == set(map(id, observations))
        assert state.pending_tools_by_name.get("web_extract") is None

    def test_explicit_tool_call_id_uses_tools_dict(self, monkeypatch):
        """When tool_call_id is present, pending_tools_by_name is not touched."""
        mod = self._make_mod()
        obs = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.tools["call-99"] = obs

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="my_tool", args={}, result='{"status": "done"}',
            task_id="task-1", session_id="sess-1", tool_call_id="call-99",
        )

        assert ended["obs"] is obs
        assert ended["output"] == {"status": "done"}
        assert not state.tools


class TestTurnScopedTraceLifecycle:
    def _make_mod(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    def test_trace_key_prefers_turn_id_without_breaking_legacy_task_key(self):
        mod = self._make_mod()
        assert mod._trace_key(task_id="task-1", session_id="sess-1", turn_id="turn-1") == "turn:turn-1"
        assert mod._trace_key(task_id="task-1", session_id="sess-1") == "task:task-1"
        assert mod._trace_key(session_id="sess-1") == "session:sess-1"
        assert mod._trace_key_candidates(
            task_id="task-1", session_id="sess-1", turn_id="turn-1"
        ) == ["turn:turn-1", "task:task-1", "task-1", "session:sess-1"]

    def test_start_root_trace_uses_turn_seed_name_and_metadata(self):
        mod = self._make_mod()

        class _Ctx:
            def __init__(self, span):
                self.span = span
                self.exited = False
            def __enter__(self):
                return self.span
            def __exit__(self, exc_type, exc, tb):
                self.exited = True

        class _Span:
            def __init__(self):
                self.trace_input = None
            def set_trace_io(self, *, input=None, output=None):
                self.trace_input = input
            def start_observation(self, **kwargs):
                return object()
            def update(self, **kwargs):
                pass
            def end(self):
                pass

        class _Client:
            def __init__(self):
                self.seed = None
                self.kwargs = None
                self.span = _Span()
            def create_trace_id(self, *, seed):
                self.seed = seed
                return "trace-id"
            def start_as_current_observation(self, **kwargs):
                self.kwargs = kwargs
                return _Ctx(self.span)

        client = _Client()
        state = mod._start_root_trace(
            "turn:turn-1",
            task_id="task-1",
            session_id="sess-1",
            turn_id="turn-1",
            platform="discord",
            provider="provider-x",
            model="model-y",
            api_mode="responses",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

        assert client.seed == "turn-1"
        assert client.kwargs["name"] == "Hermes discord turn"
        assert client.kwargs["trace_context"]["session_id"] == "sess-1"
        assert client.kwargs["metadata"]["session_id"] == "sess-1"
        assert client.kwargs["metadata"]["turn_id"] == "turn-1"
        assert state.session_id == "sess-1"
        assert state.turn_id == "turn-1"

    def test_post_llm_can_close_legacy_task_key_when_turn_id_is_present(self, monkeypatch):
        mod = self._make_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())
        generation = object()
        state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None, session_id="sess-1")
        state.generations[mod._request_key(1)] = generation
        legacy_key = "task-1"
        monkeypatch.setitem(mod._TRACE_STATE, legacy_key, state)

        ended = {}
        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output
        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_llm_call(
            task_id="task-1",
            session_id="sess-1",
            turn_id="turn-1",
            api_call_count=1,
            assistant_content_chars=0,
        )

        assert ended["obs"] is generation
        assert legacy_key in mod._TRACE_STATE

    def test_tool_post_can_find_turn_scoped_state(self, monkeypatch):
        mod = self._make_mod()
        obs = object()
        state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=None, session_id="sess-1", turn_id="turn-1")
        state.tools["call-1"] = obs
        monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key(task_id="task-1", session_id="sess-1", turn_id="turn-1"), state)

        ended = {}
        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output
        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="my_tool",
            args={},
            result='{"ok": true}',
            task_id="task-1",
            session_id="sess-1",
            turn_id="turn-1",
            tool_call_id="call-1",
        )

        assert ended["obs"] is obs
        assert ended["output"] == {"ok": True}
        assert not state.tools

    def test_session_end_closes_unfinished_trace_for_matching_session(self, monkeypatch):
        mod = self._make_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

        class _Span:
            def __init__(self):
                self.ended = False
                self.output = None
            def set_trace_io(self, *, input=None, output=None):
                self.output = output
            def update(self, **kwargs):
                self.output = kwargs.get("output")
            def end(self):
                self.ended = True

        class _Ctx:
            def __init__(self):
                self.exited = False
            def __exit__(self, exc_type, exc, tb):
                self.exited = True

        span = _Span()
        ctx = _Ctx()
        state = mod.TraceState(trace_id="trace", root_ctx=ctx, root_span=span, session_id="sess-1", turn_id="turn-1")
        key = mod._trace_key(task_id="task-1", session_id="sess-1", turn_id="turn-1")
        monkeypatch.setitem(mod._TRACE_STATE, key, state)

        mod.on_session_end(session_id="sess-1", task_id="task-1", turn_id="turn-1")

        assert key not in mod._TRACE_STATE
        assert span.ended is True
        assert ctx.exited is True
        assert span.output == {"finalized_by": "on_session_end"}

    def test_session_end_with_turn_id_does_not_close_other_turn_same_session(self, monkeypatch):
        mod = self._make_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

        class _Span:
            def __init__(self):
                self.ended = False
            def set_trace_io(self, *, input=None, output=None):
                pass
            def update(self, **kwargs):
                pass
            def end(self):
                self.ended = True

        class _Ctx:
            def __init__(self):
                self.exited = False
            def __exit__(self, exc_type, exc, tb):
                self.exited = True

        span_a, span_b = _Span(), _Span()
        ctx_a, ctx_b = _Ctx(), _Ctx()
        state_a = mod.TraceState(
            trace_id="trace-a", root_ctx=ctx_a, root_span=span_a,
            task_id="task-a", session_id="sess-1", turn_id="turn-a",
        )
        state_b = mod.TraceState(
            trace_id="trace-b", root_ctx=ctx_b, root_span=span_b,
            task_id="task-b", session_id="sess-1", turn_id="turn-b",
        )
        key_a = mod._trace_key(task_id="task-a", session_id="sess-1", turn_id="turn-a")
        key_b = mod._trace_key(task_id="task-b", session_id="sess-1", turn_id="turn-b")
        monkeypatch.setitem(mod._TRACE_STATE, key_a, state_a)
        monkeypatch.setitem(mod._TRACE_STATE, key_b, state_b)

        mod.on_session_end(session_id="sess-1", task_id="task-a", turn_id="turn-a")

        assert key_a not in mod._TRACE_STATE
        assert mod._TRACE_STATE[key_b] is state_b
        assert span_a.ended is True
        assert ctx_a.exited is True
        assert span_b.ended is False
        assert ctx_b.exited is False

    def test_finish_trace_exits_root_context_even_when_root_span_end_raises(self, monkeypatch):
        mod = self._make_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())

        class _Span:
            def __init__(self):
                self.ended = False
            def set_trace_io(self, *, input=None, output=None):
                pass
            def update(self, **kwargs):
                pass
            def end(self):
                self.ended = True
                raise RuntimeError("boom")

        class _Ctx:
            def __init__(self):
                self.exited = False
            def __exit__(self, exc_type, exc, tb):
                self.exited = True

        span = _Span()
        ctx = _Ctx()
        state = mod.TraceState(trace_id="trace", root_ctx=ctx, root_span=span, session_id="sess-1", turn_id="turn-1")
        key = mod._trace_key(task_id="task-1", session_id="sess-1", turn_id="turn-1")
        monkeypatch.setitem(mod._TRACE_STATE, key, state)

        mod._finish_trace(key, output={"ok": True})

        assert key not in mod._TRACE_STATE
        assert span.ended is True
        assert ctx.exited is True

    def test_session_end_does_not_close_other_active_sessions(self, monkeypatch):
        mod = self._make_mod()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())
        state = mod.TraceState(trace_id="trace", root_ctx=None, root_span=object(), session_id="other")
        key = mod._trace_key(task_id="task-2", session_id="other")
        monkeypatch.setitem(mod._TRACE_STATE, key, state)

        mod.on_session_end(session_id="sess-1")

        assert mod._TRACE_STATE[key] is state


class TestUsageFromSanitizedResponse:
    """Regression: ``post_api_request`` delivers ``response`` as a sanitized
    dict (no ``.usage`` attribute) plus a separate ``usage`` summary dict. The
    post-call handler must read the ``usage`` dict instead of treating the dict
    response as a usage-bearing object and dropping all token/cost data."""

    def _setup(self, mod, monkeypatch):
        # Active client so on_post_llm_call does not early-return.
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())
        observation = object()
        state = mod.TraceState(trace_id="trace-1", root_ctx=None, root_span=None)
        state.generations[mod._request_key(1)] = observation
        monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task-1", "session-1"), state)
        captured = {}

        def fake_end_observation(obs, *, output=None, metadata=None, usage_details=None, cost_details=None):
            captured["usage_details"] = usage_details

        monkeypatch.setattr(mod, "_end_observation", fake_end_observation)
        return captured

    def test_sanitized_dict_response_uses_usage_dict(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")
        captured = self._setup(mod, monkeypatch)

        # A plain dict has no ``.usage`` attribute — mirrors post_api_request.
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            api_call_count=1,
            model="gemini-3-flash-preview",
            response={"model": "gemini-3-flash-preview", "usage": {"input_tokens": 100, "output_tokens": 20}},
            usage={"input_tokens": 100, "output_tokens": 20},
            assistant_content_chars=42,
        )

        # Before the fix the dict response shadowed the usage dict and tokens
        # were lost (usage_details == {}).
        assert captured["usage_details"] == {"input": 100, "output": 20}

    def test_real_response_object_with_usage_still_used(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")
        captured = self._setup(mod, monkeypatch)

        # A response object that genuinely carries usage must still take the
        # response-object path (post_llm_call / legacy behavior).
        seen = {}

        def fake_usage_and_cost(resp, **_):
            seen["resp"] = resp
            return {"input": 7, "output": 3}, {}

        monkeypatch.setattr(mod, "_usage_and_cost", fake_usage_and_cost)

        class _Resp:
            usage = {"prompt_tokens": 7, "completion_tokens": 3}

        resp = _Resp()
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            api_call_count=1,
            model="gemini-3-flash-preview",
            response=resp,
            usage={"input_tokens": 999, "output_tokens": 999},
            assistant_content_chars=42,
        )

        assert seen["resp"] is resp
        assert captured["usage_details"] == {"input": 7, "output": 3}
