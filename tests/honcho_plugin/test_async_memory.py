"""Tests for the async-memory Honcho improvements.

Covers:
  - write_frequency parsing (async / turn / session / int)
  - resolve_session_name with session_title
  - HonchoSessionManager.save() routing per write_frequency
  - async writer thread lifecycle and retry
  - flush_all() drains pending messages
  - shutdown() joins the thread
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(**kwargs) -> HonchoSession:
    return HonchoSession(
        key=kwargs.get("key", "cli:test"),
        user_peer_id=kwargs.get("user_peer_id", "eri"),
        assistant_peer_id=kwargs.get("assistant_peer_id", "hermes"),
        honcho_session_id=kwargs.get("honcho_session_id", "cli-test"),
        messages=kwargs.get("messages", []),
    )


def _make_manager(write_frequency="turn") -> HonchoSessionManager:
    cfg = HonchoClientConfig(
        write_frequency=write_frequency,
        api_key="test-key",
        enabled=True,
    )
    mgr = HonchoSessionManager(config=cfg)
    mgr._honcho = MagicMock()
    return mgr


class _FakePeer:
    def __init__(self, peer_id: str):
        self.peer_id = peer_id

    def message(self, content: str):
        return SimpleNamespace(peer_id=self.peer_id, content=content)


def _make_manager_with_flush_fakes(message_noise_filters=None):
    cfg = HonchoClientConfig(
        write_frequency="turn",
        api_key="test-key",
        enabled=True,
        message_noise_filters=message_noise_filters or [],
    )
    mgr = HonchoSessionManager(config=cfg)
    session = _make_session(
        key="cli:test",
        user_peer_id="eri",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )
    mgr._cache[session.key] = session
    mgr._peers_cache[session.user_peer_id] = _FakePeer(session.user_peer_id)
    mgr._peers_cache[session.assistant_peer_id] = _FakePeer(session.assistant_peer_id)
    honcho_session = MagicMock()
    mgr._sessions_cache[session.honcho_session_id] = honcho_session
    return mgr, session, honcho_session


# ---------------------------------------------------------------------------
# write_frequency parsing from config file
# ---------------------------------------------------------------------------

class TestWriteFrequencyParsing:
    def test_string_async(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "async"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "async"

    def test_string_turn(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "turn"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "turn"

    def test_string_session(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "session"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "session"

    def test_integer_frequency(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": 5}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == 5

    def test_integer_string_coerced(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k", "writeFrequency": "3"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == 3

    def test_host_block_overrides_root(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "writeFrequency": "turn",
            "hosts": {"hermes": {"writeFrequency": "session"}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "session"

    def test_message_noise_filters_parse_root_strings_and_objects(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "messageNoiseFilters": [
                "startup smoke",
                {"name": "task id", "pattern": "^t_[0-9a-f]{8}$"},
                {"regex": "^proc_.*$"},
            ],
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.message_noise_filters == [
            "startup smoke",
            "^t_[0-9a-f]{8}$",
            "^proc_.*$",
        ]

    def test_message_noise_filters_host_block_overrides_root(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "apiKey": "k",
            "messageNoiseFilters": ["root-filter"],
            "hosts": {"hermes": {"messageNoiseFilters": ["host-filter"]}},
        }))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.message_noise_filters == ["host-filter"]

    def test_defaults_to_async(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"apiKey": "k"}))
        cfg = HonchoClientConfig.from_global_config(config_path=cfg_file)
        assert cfg.write_frequency == "async"


# ---------------------------------------------------------------------------
# resolve_session_name with session_title
# ---------------------------------------------------------------------------

class TestResolveSessionNameTitle:
    def test_manual_override_beats_title(self):
        cfg = HonchoClientConfig(sessions={"/my/project": "manual-name"})
        result = cfg.resolve_session_name("/my/project", session_title="the-title")
        assert result == "manual-name"

    def test_title_beats_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="my-project")
        assert result == "my-project"

    def test_title_with_peer_prefix(self):
        cfg = HonchoClientConfig(peer_name="eri", session_peer_prefix=True)
        result = cfg.resolve_session_name("/some/dir", session_title="aeris")
        assert result == "eri-aeris"

    def test_title_sanitized(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="my project/name!")
        # trailing dashes stripped by .strip('-')
        assert result == "my-project-name"

    def test_title_all_invalid_chars_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="!!! ###")
        # sanitized to empty → falls back to dirname
        assert result == "dir"

    def test_none_title_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title=None)
        assert result == "dir"

    def test_empty_title_falls_back_to_dirname(self):
        cfg = HonchoClientConfig()
        result = cfg.resolve_session_name("/some/dir", session_title="")
        assert result == "dir"

    def test_per_session_uses_session_id(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_per_session_with_peer_prefix(self):
        cfg = HonchoClientConfig(session_strategy="per-session", peer_name="eri", session_peer_prefix=True)
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "eri-20260309_175514_9797dd"

    def test_per_session_no_id_falls_back_to_dirname(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_id=None)
        assert result == "dir"

    def test_per_session_id_beats_title(self):
        # per-session: the run's session_id is authoritative; an (auto-)generated
        # title must NOT remap a live conversation onto a second Honcho session.
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name("/some/dir", session_title="my-title", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_per_session_id_beats_manual_map(self):
        # per-session: session_id also wins over a stale cwd map entry (e.g. the
        # desktop launching from a mapped home dir).
        cfg = HonchoClientConfig(session_strategy="per-session", sessions={"/some/dir": "pinned"})
        result = cfg.resolve_session_name("/some/dir", session_id="20260309_175514_9797dd")
        assert result == "20260309_175514_9797dd"

    def test_title_still_applies_for_non_per_session(self):
        # Outside per-session, /title still names the Honcho session.
        cfg = HonchoClientConfig(session_strategy="per-directory")
        result = cfg.resolve_session_name("/some/dir", session_title="my-title", session_id="20260309_175514_9797dd")
        assert result == "my-title"

    def test_gateway_key_beats_title_for_gateway_sessions(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name(
            "/some/dir",
            session_title="pretty title",
            session_id="20260309_175514_9797dd",
            gateway_session_key="webui:session:abc123",
        )
        assert result == "webui-session-abc123"

    def test_gateway_key_with_peer_prefix(self):
        cfg = HonchoClientConfig(
            session_strategy="per-session",
            peer_name="eri",
            session_peer_prefix=True,
        )
        result = cfg.resolve_session_name(
            "/some/dir",
            session_title="pretty title",
            session_id="20260309_175514_9797dd",
            gateway_session_key="webui:session:abc123",
        )
        assert result == "eri-webui-session-abc123"

    def test_gateway_key_beats_manual_map_and_title(self):
        cfg = HonchoClientConfig(
            session_strategy="per-session",
            sessions={"/some/dir": "manual-name"},
        )
        result = cfg.resolve_session_name(
            "/some/dir",
            session_title="pretty title",
            session_id="20260309_175514_9797dd",
            gateway_session_key="webui:session:abc123",
        )
        assert result == "webui-session-abc123"

    def test_invalid_gateway_key_falls_back_to_per_session_id(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        result = cfg.resolve_session_name(
            "/some/dir",
            session_title="pretty title",
            session_id="20260309_175514_9797dd",
            gateway_session_key="!!! ###",
        )
        assert result == "20260309_175514_9797dd"

    def test_invalid_gateway_key_falls_back_to_title_for_non_per_session(self):
        cfg = HonchoClientConfig(session_strategy="per-directory")
        result = cfg.resolve_session_name(
            "/some/dir",
            session_title="pretty title",
            gateway_session_key="!!! ###",
        )
        assert result == "pretty-title"

    def test_overlong_gateway_key_with_peer_prefix_is_limited(self):
        cfg = HonchoClientConfig(
            session_strategy="per-session",
            peer_name="eri",
            session_peer_prefix=True,
        )
        result = cfg.resolve_session_name(
            "/some/dir",
            gateway_session_key="webui:" + "x" * 180,
        )
        assert result.startswith("eri-webui-")
        assert len(result) <= cfg._HONCHO_SESSION_ID_MAX_LEN

    def test_session_name_candidates_include_gateway_primary_and_legacy_title(self):
        cfg = HonchoClientConfig(session_strategy="per-session")
        candidates = cfg.resolve_session_name_candidates(
            "/some/dir",
            session_title="pretty title",
            session_id="20260309_175514_9797dd",
            gateway_session_key="webui:session:abc123",
        )
        assert candidates[0] == ("gateway_session_key", "webui-session-abc123")
        assert ("session_title", "pretty-title") in candidates
        assert ("session_id", "20260309_175514_9797dd") in candidates

    def test_global_strategy_returns_workspace(self):
        cfg = HonchoClientConfig(session_strategy="global", workspace_id="my-workspace")
        result = cfg.resolve_session_name("/some/dir")
        assert result == "my-workspace"


# ---------------------------------------------------------------------------
# save() routing per write_frequency
# ---------------------------------------------------------------------------

class TestSaveRouting:
    def _make_session_with_message(self, mgr=None):
        sess = _make_session()
        sess.add_message("user", "hello")
        sess.add_message("assistant", "hi")
        if mgr:
            mgr._cache[sess.key] = sess
        return sess

    def test_turn_flushes_immediately(self):
        mgr = _make_manager(write_frequency="turn")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_called_once_with(sess)

    def test_session_mode_does_not_flush(self):
        mgr = _make_manager(write_frequency="session")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            mock_flush.assert_not_called()

    def test_async_mode_enqueues(self):
        mgr = _make_manager(write_frequency="async")
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)
            # flush_session should NOT be called synchronously
            mock_flush.assert_not_called()
        assert not mgr._async_queue.empty()

    def test_int_frequency_flushes_on_nth_turn(self):
        mgr = _make_manager(write_frequency=3)
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.save(sess)  # turn 1
            mgr.save(sess)  # turn 2
            assert mock_flush.call_count == 0
            mgr.save(sess)  # turn 3
            assert mock_flush.call_count == 1

    def test_int_frequency_skips_other_turns(self):
        mgr = _make_manager(write_frequency=5)
        sess = self._make_session_with_message(mgr)
        with patch.object(mgr, "_flush_session") as mock_flush:
            for _ in range(4):
                mgr.save(sess)
            assert mock_flush.call_count == 0
            mgr.save(sess)  # turn 5
            assert mock_flush.call_count == 1


class _CaptureSyncManager:
    def __init__(self):
        self.requested_session_keys = []
        self.flushed_session_keys = []
        self.sessions = {}

    def get_or_create(self, session_key: str):
        self.requested_session_keys.append(session_key)
        if session_key not in self.sessions:
            self.sessions[session_key] = _make_session(
                key=session_key,
                honcho_session_id=session_key,
            )
        return self.sessions[session_key]

    def _flush_session(self, session):
        self.flushed_session_keys.append(session.key)
        return True


class TestHonchoProviderSessionSwitch:
    def _make_ready_provider(self) -> tuple[HonchoMemoryProvider, _CaptureSyncManager]:
        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(
            session_strategy="per-session",
            write_frequency="turn",
            enabled=True,
            api_key="test-key",
        )
        manager = _CaptureSyncManager()
        provider._manager = manager
        provider._session_initialized = True
        provider._session_key = provider._resolve_session_key(provider._config, "old-session")
        return provider, manager

    def test_session_switch_updates_cached_key_for_later_sync_turns(self):
        provider, manager = self._make_ready_provider()

        provider.sync_turn("before", "old response")
        provider._sync_thread.join(timeout=2)

        provider.on_session_switch(
            "new-session",
            parent_session_id="old-session",
            reset=False,
        )
        provider.sync_turn("after", "new response")
        provider._sync_thread.join(timeout=2)

        assert manager.requested_session_keys == ["old-session", "new-session"]
        assert manager.flushed_session_keys == ["old-session", "new-session"]
        assert provider._session_key == "new-session"

    def test_session_switch_prefers_gateway_key_over_title(self):
        provider, manager = self._make_ready_provider()

        provider.on_session_switch(
            "new-session",
            parent_session_id="old-session",
            session_title="pretty title",
            gateway_session_key="webui:session:abc123",
            reset=False,
        )
        provider.sync_turn("after", "new response")
        provider._sync_thread.join(timeout=2)

        assert manager.requested_session_keys == ["webui-session-abc123"]
        assert manager.flushed_session_keys == ["webui-session-abc123"]
        assert provider._session_key == "webui-session-abc123"


class TestMessageNoiseFilters:
    FILTERS = [
        r"\bstartup smoke\b",
        r"\brespond\s+OK\s+only\b",
        r"^\s*t_[0-9a-f]{8}\s*$",
        r"^\s*proc_[A-Za-z0-9_:-]+\s*$",
        r"\bmessage sent at timestamp\b",
        r"\bone[- ]off command[- ]format instruction\b",
    ]

    def test_flush_filters_transient_noise_and_keeps_durable_message(self):
        mgr, session, honcho_session = _make_manager_with_flush_fakes(self.FILTERS)
        session.add_message("user", "startup smoke: respond OK only")
        session.add_message("user", "t_e7f0d4a8")
        session.add_message("assistant", "proc_7c69a48298f2")
        session.add_message("assistant", "message sent at timestamp 2026-06-21T10:00:00Z")
        session.add_message("user", "one-off command-format instruction: reply CSV for this command")
        session.add_message("user", "Janusz prefers rigorous review before memory cleanup.")

        assert mgr._flush_session(session) is True

        honcho_session.add_messages.assert_called_once()
        sent = honcho_session.add_messages.call_args.args[0]
        assert [message.content for message in sent] == [
            "Janusz prefers rigorous review before memory cleanup."
        ]
        assert all(message.get("_synced") for message in session.messages)
        assert [message.get("_filtered", False) for message in session.messages] == [
            True,
            True,
            True,
            True,
            True,
            False,
        ]

    def test_flush_marks_all_filtered_messages_synced_without_add_messages(self):
        mgr, session, honcho_session = _make_manager_with_flush_fakes(self.FILTERS)
        session.add_message("user", "respond OK only")
        session.add_message("assistant", "t_03c8f3d0")

        assert mgr._flush_session(session) is True

        honcho_session.add_messages.assert_not_called()
        assert all(message.get("_synced") for message in session.messages)
        assert all(message.get("_filtered") for message in session.messages)

    def test_invalid_noise_filter_is_ignored_fail_open(self):
        mgr, session, honcho_session = _make_manager_with_flush_fakes(["["])
        session.add_message("user", "startup smoke: respond OK only")

        assert mgr._flush_session(session) is True

        honcho_session.add_messages.assert_called_once()
        sent = honcho_session.add_messages.call_args.args[0]
        assert [message.content for message in sent] == ["startup smoke: respond OK only"]


# ---------------------------------------------------------------------------
# flush_all()
# ---------------------------------------------------------------------------

class TestFlushAll:
    def test_flushes_all_cached_sessions(self):
        mgr = _make_manager(write_frequency="session")
        s1 = _make_session(key="s1", honcho_session_id="s1")
        s2 = _make_session(key="s2", honcho_session_id="s2")
        s1.add_message("user", "a")
        s2.add_message("user", "b")
        mgr._cache = {"s1": s1, "s2": s2}

        with patch.object(mgr, "_flush_session") as mock_flush:
            mgr.flush_all()
            assert mock_flush.call_count == 2

    def test_flush_all_drains_async_queue(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "pending")

        with patch.object(mgr, "_flush_session") as mock_flush:
            # Put the item AFTER the mock is installed so the background
            # writer thread (if it dequeues before flush_all) still hits
            # the mock rather than the real _flush_session.
            mgr._async_queue.put(sess)
            mgr.flush_all()
            # Called at least once for the queued item
            assert mock_flush.call_count >= 1

    def test_flush_all_tolerates_errors(self):
        mgr = _make_manager(write_frequency="session")
        sess = _make_session()
        mgr._cache = {"key": sess}
        with patch.object(mgr, "_flush_session", side_effect=RuntimeError("oops")):
            # Should not raise
            mgr.flush_all()


# ---------------------------------------------------------------------------
# async writer thread lifecycle
# ---------------------------------------------------------------------------

class TestAsyncWriterThread:
    def test_thread_started_on_async_mode(self):
        mgr = _make_manager(write_frequency="async")
        assert mgr._async_thread is not None
        assert mgr._async_thread.is_alive()
        mgr.shutdown()

    def test_no_thread_for_turn_mode(self):
        mgr = _make_manager(write_frequency="turn")
        assert mgr._async_thread is None
        assert mgr._async_queue is None

    def test_shutdown_joins_thread(self):
        mgr = _make_manager(write_frequency="async")
        assert mgr._async_thread.is_alive()
        mgr.shutdown()
        assert not mgr._async_thread.is_alive()

    def test_async_writer_calls_flush(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "async msg")

        flushed = []
        flushed_event = threading.Event()

        def capture(session):
            flushed.append(session)
            flushed_event.set()
            return True

        mgr._flush_session = capture
        mgr._async_queue.put(sess)
        assert flushed_event.wait(timeout=10), "async writer never flushed"

        mgr.shutdown()
        assert len(flushed) == 1
        assert flushed[0] is sess

    def test_shutdown_sentinel_stops_loop(self):
        mgr = _make_manager(write_frequency="async")
        thread = mgr._async_thread
        mgr.shutdown()
        thread.join(timeout=10)
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# async retry on failure
# ---------------------------------------------------------------------------

class TestAsyncWriterRetry:
    def test_retries_once_on_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def flaky_flush(session):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("network blip")
            retry_done.set()
            return True

        mgr._flush_session = flaky_flush

        with patch("time.sleep"):  # skip the 2s sleep in retry
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        assert call_count[0] == 2

    def test_drops_after_two_failures(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def always_fail(session):
            call_count[0] += 1
            if call_count[0] >= 2:
                retry_done.set()
            raise RuntimeError("always broken")

        mgr._flush_session = always_fail

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        # Should have tried exactly twice (initial + one retry) and not crashed
        assert call_count[0] == 2
        assert not mgr._async_thread.is_alive()

    def test_retries_when_flush_reports_failure(self):
        mgr = _make_manager(write_frequency="async")
        sess = _make_session()
        sess.add_message("user", "msg")

        call_count = [0]
        retry_done = threading.Event()

        def fail_then_succeed(session):
            call_count[0] += 1
            if call_count[0] >= 2:
                retry_done.set()
            return call_count[0] > 1

        mgr._flush_session = fail_then_succeed

        with patch("time.sleep"):
            mgr._async_queue.put(sess)
            assert retry_done.wait(timeout=10), "async writer never retried"

        mgr.shutdown()
        assert call_count[0] == 2


class TestMemoryFileMigrationTargets:
    def test_soul_upload_targets_ai_peer(self, tmp_path):
        mgr = _make_manager(write_frequency="turn")
        session = _make_session(
            key="cli:test",
            user_peer_id="custom-user",
            assistant_peer_id="custom-ai",
            honcho_session_id="cli-test",
        )
        mgr._cache[session.key] = session

        user_peer = MagicMock(name="user-peer")
        ai_peer = MagicMock(name="ai-peer")
        mgr._peers_cache[session.user_peer_id] = user_peer
        mgr._peers_cache[session.assistant_peer_id] = ai_peer

        honcho_session = MagicMock()
        mgr._sessions_cache[session.honcho_session_id] = honcho_session

        (tmp_path / "MEMORY.md").write_text("memory facts", encoding="utf-8")
        (tmp_path / "USER.md").write_text("user profile", encoding="utf-8")
        (tmp_path / "SOUL.md").write_text("ai identity", encoding="utf-8")

        uploaded = mgr.migrate_memory_files(session.key, str(tmp_path))

        assert uploaded is True
        assert honcho_session.upload_file.call_count == 3

        peer_by_upload_name = {}
        for call_args in honcho_session.upload_file.call_args_list:
            payload = call_args.kwargs["file"]
            peer_by_upload_name[payload[0]] = call_args.kwargs["peer"]

        assert peer_by_upload_name["consolidated_memory.md"] is user_peer
        assert peer_by_upload_name["user_profile.md"] is user_peer
        assert peer_by_upload_name["agent_soul.md"] is ai_peer


# ---------------------------------------------------------------------------
# HonchoClientConfig dataclass defaults for new fields
# ---------------------------------------------------------------------------

class TestNewConfigFieldDefaults:
    def test_write_frequency_default(self):
        cfg = HonchoClientConfig()
        assert cfg.write_frequency == "async"

    def test_write_frequency_set(self):
        cfg = HonchoClientConfig(write_frequency="turn")
        assert cfg.write_frequency == "turn"


class TestPrefetchCacheAccessors:
    def test_set_and_pop_context_result(self):
        mgr = _make_manager(write_frequency="turn")
        payload = {"representation": "Known user", "card": "prefers concise replies"}

        mgr.set_context_result("cli:test", payload)

        assert mgr.pop_context_result("cli:test") == payload
        assert mgr.pop_context_result("cli:test") == {}

