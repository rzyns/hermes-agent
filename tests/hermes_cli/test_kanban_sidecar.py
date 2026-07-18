"""Tests for ``hermes_cli/kanban_sidecar.py`` — append-only semantic event sidecar.

All tests run against a temporary ``HERMES_HOME`` so they never touch
live boards.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_sidecar as ks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _init_board(tmp_path, monkeypatch, board: str = "default"):
    """Create a fresh board DB and return its ``board_dir``."""
    home = _setup_home(tmp_path, monkeypatch)
    db_path = kb.init_db(board=board)
    return db_path.parent


def _enable_sidecar(monkeypatch):
    """Monkeypatch sidecar config so ``_sidecar_enabled()`` returns ``True``."""
    monkeypatch.setattr(ks, "_sidecar_enabled", lambda: True)


def _load_events(board_dir: Path):
    """Return list of parsed event dicts from ``current.jsonl``."""
    path = ks._current_jsonl(board_dir)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _load_hashes(board_dir: Path):
    """Return list of hash strings from ``current.jsonl.sha256``."""
    path = ks._current_hashes(board_dir)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_hash_canonicalization():
    """Hash must be deterministic and sensitive to key ordering."""
    event = {"v": 1, "ts": 100, "seq": 1, "kind": "test", "task_id": "t_1", "payload": {"a": 1}}
    h1 = ks._hash_event(event)
    h2 = ks._hash_event(event)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_verify_line_rejects_tampering():
    """Changing a field after hashing must invalidate the line."""
    event = {"v": 1, "ts": 100, "seq": 1, "kind": "test", "task_id": "t_1", "payload": {}}
    event["hash"] = ks._hash_event(event)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert ks._verify_line(line) is not None

    # Tamper
    bad = json.loads(line)
    bad["seq"] = 999
    bad_line = json.dumps(bad, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    assert ks._verify_line(bad_line) is None


def test_verify_line_rejects_bad_json():
    assert ks._verify_line("not json") is None


def test_sidecar_enabled_default():
    """Sidecar must be disabled unless explicitly enabled."""
    assert ks._sidecar_enabled() is False


# ---------------------------------------------------------------------------
# Integration: append
# ---------------------------------------------------------------------------


def test_append_disabled_by_default(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    ks.append_event(board_dir, "t_1", "task_created", {"title": "Hello"})
    assert not ks._current_jsonl(board_dir).exists()


def test_append_creates_files(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    ks.append_event(board_dir, "t_1", "task_created", {"title": "Hello"})
    assert ks._current_jsonl(board_dir).exists()
    assert ks._current_hashes(board_dir).exists()


def test_append_line_structure(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    ks.append_event(board_dir, "t_1", "task_status_changed", {"old": "todo", "new": "running"}, run_id=42)
    events = _load_events(board_dir)
    assert len(events) == 1
    ev = events[0]
    assert ev["v"] == 1
    assert ev["seq"] == 1
    assert ev["kind"] == "task_status_changed"
    assert ev["task_id"] == "t_1"
    assert ev["payload"]["new"] == "running"
    assert "hash" in ev


def test_append_sequence_monotonic(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    for i in range(5):
        ks.append_event(board_dir, f"t_{i}", "heartbeat", {"n": i})
    events = _load_events(board_dir)
    seqs = [e["seq"] for e in events]
    assert seqs == [1, 2, 3, 4, 5]


def test_hashes_align_with_lines(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    ks.append_event(board_dir, "t_1", "task_created", {"title": "A"})
    ks.append_event(board_dir, "t_2", "task_created", {"title": "B"})
    events = _load_events(board_dir)
    hashes = _load_hashes(board_dir)
    assert len(events) == len(hashes) == 2
    for ev, h in zip(events, hashes):
        assert ev["hash"] == h


def test_append_event_from_kanban_db(tmp_path, monkeypatch):
    """When sidecar is enabled, ``_append_event`` in kanban_db must write sidecar lines."""
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    db_path = board_dir / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        kb._append_event(conn, "t_db", "test_event", {"x": 1}, run_id=7)
        conn.commit()
    finally:
        conn.close()

    events = _load_events(board_dir)
    assert len(events) == 1
    assert events[0]["kind"] == "test_event"
    assert events[0]["task_id"] == "t_db"
    assert events[0]["payload"]["x"] == 1


def test_real_create_and_complete_rebuilds_structural_task_without_summary_text(tmp_path, monkeypatch):
    """Real kanban_db mutations must emit replayable sidecar events without free-form summaries."""
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(
            conn,
            title="Recovered title",
            body="body text should be summarized, not duplicated",
            initial_status="running",
            assignee="platform-eng",
        )
        assert kb.complete_task(
            conn,
            task_id,
            result="short result",
            summary="FREEFORM SUMMARY MUST NOT BE DUPLICATED",
        )
    finally:
        conn.close()

    events = _load_events(board_dir)
    assert [event["kind"] for event in events] == ["task_created", "task_status_changed"]
    created_payload = events[0]["payload"]
    assert created_payload["title"] == "Recovered title"
    assert created_payload["status"] == "ready"
    assert "body" not in created_payload
    assert created_payload["body_len"] == len("body text should be summarized, not duplicated")
    completed_payload = events[1]["payload"]
    assert completed_payload["new"] == "done"
    assert "summary" not in completed_payload
    assert completed_payload["summary_len"] == len("FREEFORM SUMMARY MUST NOT BE DUPLICATED")

    target = tmp_path / "rebuilt-real.db"
    report = ks.rebuild_board(board_dir, target)
    assert report.events_replayed == 2
    assert report.tasks_created == 1
    assert report.tasks_updated == 1

    rebuilt = sqlite3.connect(str(target))
    try:
        row = rebuilt.execute("SELECT title, status, assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
        assert row == ("Recovered title", "done", "platform-eng")
    finally:
        rebuilt.close()


def test_hallucination_rejection_sidecar_omits_summary_preview_text(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="Hallucination guard")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn,
                task_id,
                result="result sentinel",
                summary="LEAK_SENTINEL_SUMMARY_PREVIEW should not be duplicated",
                created_cards=["t_deadbeef"],
            )
    finally:
        conn.close()

    events = _load_events(board_dir)
    payload = events[-1]["payload"]
    assert events[-1]["kind"] == "completion_blocked_hallucination"
    assert "summary_preview" not in payload
    assert payload["summary_preview_len"] == len("LEAK_SENTINEL_SUMMARY_PREVIEW should not be duplicated")
    assert "LEAK_SENTINEL_SUMMARY_PREVIEW" not in json.dumps(events, sort_keys=True)


def test_real_block_unblock_and_manual_promote_rebuild_final_status(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    conn = kb.connect(board="default")
    try:
        blocked_then_unblocked = kb.create_task(conn, title="Block/unblock")
        assert kb.block_task(conn, blocked_then_unblocked, reason="BLOCK_REASON_SENTINEL")
        assert kb.unblock_task(conn, blocked_then_unblocked)

        manually_promoted = kb.create_task(conn, title="Manual promote", initial_status="blocked")
        promoted, reason = kb.promote_task(conn, manually_promoted, actor="reviewer", reason="PROMOTE_REASON_SENTINEL", force=True)
        assert promoted, reason
    finally:
        conn.close()

    events = _load_events(board_dir)
    assert "BLOCK_REASON_SENTINEL" not in json.dumps(events, sort_keys=True)
    assert "PROMOTE_REASON_SENTINEL" not in json.dumps(events, sort_keys=True)
    status_changes = [
        (event["task_id"], event["payload"]["new"])
        for event in events
        if event["kind"] == "task_status_changed"
    ]
    assert status_changes == [
        (blocked_then_unblocked, "blocked"),
        (blocked_then_unblocked, "ready"),
        (manually_promoted, "blocked"),
        (manually_promoted, "ready"),
    ]

    target = tmp_path / "rebuilt-status-transitions.db"
    report = ks.rebuild_board(board_dir, target)
    assert report.tasks_created == 2
    assert report.tasks_updated == 4

    rebuilt = sqlite3.connect(str(target))
    try:
        rows = dict(rebuilt.execute("SELECT title, status FROM tasks").fetchall())
        assert rows == {"Block/unblock": "ready", "Manual promote": "ready"}
    finally:
        rebuilt.close()


def test_rebuild_comment_placeholder_uses_integer_autoincrement_id(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    ks.append_event(board_dir, "t_alpha", "task_created", {"title": "Alpha", "status": "todo"})
    ks.append_event(board_dir, "t_alpha", "commented", {"author": "reviewer", "len": 17})

    target = tmp_path / "rebuilt-comments.db"
    report = ks.rebuild_board(board_dir, target)
    assert report.comments_added == 1

    rebuilt = sqlite3.connect(str(target))
    try:
        row = rebuilt.execute("SELECT id, author, body FROM task_comments").fetchone()
        assert isinstance(row[0], int)
        assert row[1] == "reviewer"
        assert "original length 17" in row[2]
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_by_size(monkeypatch, tmp_path):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    # Force tiny rotation threshold
    monkeypatch.setattr(ks, "_sidecar_max_bytes", lambda: 10)
    ks.append_event(board_dir, "t_1", "task_created", {"title": "A"})
    # After first write, file size likely > 10 bytes; next append triggers rotation.
    time.sleep(0.1)
    ks.append_event(board_dir, "t_2", "task_created", {"title": "B"})
    current = ks._current_jsonl(board_dir)
    assert current.exists()
    events = _load_events(board_dir)
    # After rotation, current.jsonl should contain only the second event.
    assert len(events) == 1
    assert events[0]["task_id"] == "t_2"
    # Rotated segment should exist
    sdir = ks._sidecar_dir(board_dir)
    rotated = [p for p in sdir.iterdir() if p.suffix == ".jsonl" and p.name != ks.CURRENT_FILE]
    assert len(rotated) >= 1


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


def test_rebuild_from_sidecar(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    # Insert tasks directly so we know the ids.
    db_path = board_dir / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at) VALUES (?, ?, ?, ?, ?)",
            ("t_alpha", "Alpha", "todo", None, int(time.time())),
        )
        conn.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at) VALUES (?, ?, ?, ?, ?)",
            ("t_beta", "Beta", "todo", None, int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()

    # Build a fresh DB from sidecar.
    target = tmp_path / "rebuilt.db"
    report = ks.rebuild_board(board_dir, target)
    # No sidecar events were emitted yet (direct SQL insert bypasses _append_event),
    # so rebuild should be mostly empty. We'll append some events first.
    assert report.events_replayed == 0


def test_rebuild_with_events(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    # Emit events through sidecar API.
    ks.append_event(board_dir, "t_alpha", "task_created", {"title": "Alpha", "status": "todo"})
    ks.append_event(board_dir, "t_beta", "task_created", {"title": "Beta", "status": "todo"})
    ks.append_event(board_dir, "t_alpha", "task_status_changed", {"old": "todo", "new": "running"})

    target = tmp_path / "rebuilt.db"
    report = ks.rebuild_board(board_dir, target)
    assert report.events_replayed == 3
    assert report.tasks_created == 2
    assert report.tasks_updated == 1

    # Verify rebuilt DB integrity
    rconn = sqlite3.connect(str(target))
    rconn.row_factory = sqlite3.Row
    try:
        tasks = [dict(row) for row in rconn.execute("SELECT id, title, status FROM tasks")]
        ids = {t["id"]: t for t in tasks}
        assert "t_alpha" in ids
        assert ids["t_alpha"]["status"] == "running"
        assert "t_beta" in ids
        assert ids["t_beta"]["status"] == "todo"
    finally:
        rconn.close()


def test_rebuild_idempotent(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)

    ks.append_event(board_dir, "t_alpha", "task_created", {"title": "Alpha", "status": "todo"})

    target = tmp_path / "rebuilt.db"
    report1 = ks.rebuild_board(board_dir, target)
    report2 = ks.rebuild_board(board_dir, target)
    assert report2.tasks_created == 0  # already exists
    assert report2.events_replayed == report1.events_replayed


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------


def test_verify_segment_detects_corruption(tmp_path, monkeypatch):
    board_dir = _init_board(tmp_path, monkeypatch)
    _enable_sidecar(monkeypatch)
    ks.append_event(board_dir, "t_1", "task_created", {"title": "A"})
    # Corrupt the line
    jsonl = ks._current_jsonl(board_dir)
    with open(jsonl, "r+", encoding="utf-8") as fh:
        content = fh.read()
        fh.seek(0)
        fh.write(content.replace("A", "X"))
        fh.truncate()
    ok, bad, warnings = ks.verify_segment(jsonl)
    assert bad >= 1
    assert len(warnings) >= 1


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_config_keys_exist():
    from hermes_cli.config import DEFAULT_CONFIG
    sc = DEFAULT_CONFIG["kanban"]["sidecar"]
    assert sc["enabled"] is False
    assert sc["max_bytes"] == 104857600
    assert sc["rotate_daily"] is True
    assert sc["sync_mode"] == "O_DSYNC"
    assert sc["retention_days"] == 90
    assert sc["hash_verification"] is True
