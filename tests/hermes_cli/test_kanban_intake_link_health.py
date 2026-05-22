"""Tests for hermes_cli/kanban_intake_link_health.py.

Covers:
- check_register_for_task with missing and present provisional/full entries
- scan_board_for_health with real DB entries
- provisional_entry_count with empty vs populated JSONL
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_intake_link as kil
from hermes_cli import kanban_intake_link_health as kih


@pytest.fixture(autouse=True)
def _tmp_env(tmp_path, monkeypatch):
    """Isolated kanban DB + artifact tree per test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield


@pytest.fixture
def conn(_tmp_env):
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intake_task_body(url: str) -> str:
    return kil.build_intake_link_body(
        url=url,
        context=None,
        note=None,
        source="cli",
        board="attention-intake",
        assignee="link-analyst",
        idempotency_key=kil.canonical_url_hash(url),
        workspace_path="/tmp/fake-artifacts/t_xyz",
    )


def _write_register_entry(home: Path, entry: dict, *, md: bool = False) -> None:
    root = home / "artifacts" / "attention-intake"
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "register.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if md:
        md_path = root / "register.md"
        md_path.write_text(f"## Register\n\n- {entry.get('url', 'entry')}\n")


# ---------------------------------------------------------------------------
# check_register_for_task
# ---------------------------------------------------------------------------


def test_check_register_missing_all(conn, tmp_path, monkeypatch):
    """Task body present but no register entries at all."""
    body = _make_intake_task_body("https://example.com/a")
    tid = kil.create_intake_link(conn, url="https://example.com/a")
    result = kih.check_register_for_task(tid, body, hermes_home=tmp_path)
    assert result["task_id"] == tid
    assert result["has_provisional_entry"] is False
    assert result["has_full_entry"] is False
    assert result["verdict"] == "missing_provisional"


def test_check_register_provisional_only(conn, tmp_path, monkeypatch):
    body = _make_intake_task_body("https://example.com/b")
    tid = kil.create_intake_link(conn, url="https://example.com/b")
    # Mock the register entry manually to avoid kanban_db logic.
    entry = {
        "event": "intake_link_created",
        "task_id": tid,
        "url": "https://example.com/b",
        "idempotency_key": kil.canonical_url_hash("https://example.com/b"),
        "created_at": "2026-05-22T12:00:00Z",
        "board": "attention-intake",
        "status": "needs_assessment",
    }
    _write_register_entry(tmp_path, entry)
    result = kih.check_register_for_task(tid, body, hermes_home=tmp_path)
    assert result["has_provisional_entry"] is True
    assert result["has_full_entry"] is False
    assert result["verdict"] == "provisional_only"


def test_check_register_complete(conn, tmp_path, monkeypatch):
    body = _make_intake_task_body("https://example.com/c")
    tid = "t_12345"
    entry = {
        "url": "https://example.com/c",
        "source_task": tid,
        "verdict": "read",
        "created_at": "2026-05-22T12:00:00Z",
    }
    # Also include a provisional entry (different row) so the flow passes provisional step.
    prov = {
        "event": "intake_link_created",
        "task_id": tid,
        "url": "https://example.com/c",
        "idempotency_key": "abc",
        "created_at": "2026-05-22T12:00:00Z",
        "board": "attention-intake",
        "status": "needs_assessment",
    }
    _write_register_entry(tmp_path, prov)
    _write_register_entry(tmp_path, entry)
    result = kih.check_register_for_task(tid, body, hermes_home=tmp_path)
    assert result["has_provisional_entry"] is True
    assert result["has_full_entry"] is True
    assert result["body_contract_ok"] is True
    assert result["verdict"] == "complete"


def test_check_register_incomplete_body(conn, tmp_path, monkeypatch):
    tid = "t_deadbeef"
    # Body missing register pointers and needs_assessment.
    result = kih.check_register_for_task(tid, "some random body", hermes_home=tmp_path)
    assert result["body_contract_ok"] is False
    assert result["verdict"] == "incomplete_body"


# ---------------------------------------------------------------------------
# scan_board_for_health
# ---------------------------------------------------------------------------


def test_scan_board_for_health_empty(conn, tmp_path, monkeypatch):
    """Scan returns empty when board has no intake-link tasks."""
    result = kih.scan_board_for_health(conn, board="default", hermes_home=tmp_path)
    assert result["scanned_task_count"] == 0
    assert result["counts"] == {
        "provisionally_registered": 0,
        "fully_registered": 0,
        "missing_provisional": 0,
        "provisional_only": 0,
        "incomplete_body": 0,
    }


def test_scan_board_for_health_with_tasks(conn, tmp_path, monkeypatch):
    tid = kil.create_intake_link(conn, url="https://example.com/x")
    result = kih.scan_board_for_health(conn, board="default", hermes_home=tmp_path)
    assert result["scanned_task_count"] == 1
    assert result["counts"]["missing_provisional"] == 1
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["task_id"] == tid


# ---------------------------------------------------------------------------
# provisional_entry_count
# ---------------------------------------------------------------------------


def test_provisional_entry_count_empty(tmp_path):
    result = kih.provisional_entry_count(hermes_home=tmp_path)
    assert result["provisional_count"] == 0
    assert result["total_rows"] == 0
    assert result["register_jsonl_exists"] is False


def test_provisional_entry_count_non_empty(tmp_path):
    _write_register_entry(
        tmp_path,
        {"event": "intake_link_created", "task_id": "t_1", "url": "https://a.com"},
    )
    _write_register_entry(tmp_path, {"url": "https://a.com", "source_task": "t_1"})
    result = kih.provisional_entry_count(hermes_home=tmp_path)
    assert result["provisional_count"] == 1
    assert result["total_rows"] == 2
    assert result["register_jsonl_exists"] is True


# ---------------------------------------------------------------------------
# Regression: health scan must surface malformed/missing-body rows
# ---------------------------------------------------------------------------


def test_scan_board_for_health_catches_body_empty_rows(conn, tmp_path, monkeypatch):
    """A row created with title 'Link drop: ...' but no body (e.g. due to
    an old default_workdir bug) must still be surfaced by health scan."""
    # Manually insert a title-only row — simulates the failure mode.
    tid = kb.create_task(
        conn,
        title="Link drop: https://example.com/broken",
        body=None,  # missing body — the old bug
        assignee="link-analyst",
        created_by="test",
        workspace_kind="scratch",
        workspace_path=None,
        initial_status="running",
    )
    result = kih.scan_board_for_health(conn, board="default", hermes_home=tmp_path)
    # The old code would have skipped this because body lacked the contract string.
    # The fixed code selects by title LIKE 'Link drop:%' so it is included.
    assert result["scanned_task_count"] == 1
    assert result["tasks"][0]["task_id"] == tid
    assert result["tasks"][0]["verdict"] == "incomplete_body"
    assert result["counts"]["incomplete_body"] == 1
