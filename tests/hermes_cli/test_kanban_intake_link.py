"""Tests for hermes_cli/kanban_intake_link.py — Attention Intake link-drop contract.

Covers:
- canonical_url_hash normalisation
- build_intake_link_body shape
- create_intake_link with fresh row and idempotency hit
- workspace_path / mkdir / register write
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_intake_link as kil


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _tmp_env(tmp_path, monkeypatch):
    """Each test gets an isolated kanban DB + artifact tree."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    yield


@pytest.fixture
def conn(_tmp_env):
    """Yield a DB connection and close it after the test."""
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------


def test_canonical_url_hash_normalises_scheme_host_port_and_whitespace():
    a = kil.canonical_url_hash("  HTTPS://example.com:443/foo  ")
    b = kil.canonical_url_hash("https://example.com/foo")
    assert a == b
    assert len(a) == 64


def test_canonical_url_hash_preserves_non_root_trailing_slash():
    assert kil.canonical_url_hash("https://example.com/foo/") != kil.canonical_url_hash("https://example.com/foo")


def test_canonical_url_hash_normalises_empty_and_root_path():
    assert kil.canonical_url_hash("https://example.com") == kil.canonical_url_hash("https://example.com/")


def test_canonical_url_hash_preserves_path_case():
    """/Foo and /foo must remain distinct."""
    assert kil.canonical_url_hash("https://example.com/Foo") != kil.canonical_url_hash("https://example.com/foo")


def test_canonical_url_hash_preserves_percent_encoding():
    """%2F and %2f are semantically different before decode; keep them distinct."""
    assert kil.canonical_url_hash("https://example.com/a%2Fb") != kil.canonical_url_hash("https://example.com/a%2fb")


def test_canonical_url_hash_preserves_query_and_fragment():
    a = kil.canonical_url_hash("https://example.com/foo?a=1")
    b = kil.canonical_url_hash("https://example.com/foo?b=1")
    assert a != b
    c = kil.canonical_url_hash("https://example.com/foo#bar")
    d = kil.canonical_url_hash("https://example.com/foo#baz")
    assert c != d


def test_canonical_url_hash_drops_default_ports():
    a = kil.canonical_url_hash("https://example.com:443/")
    b = kil.canonical_url_hash("https://example.com/")
    assert a == b
    c = kil.canonical_url_hash("http://example.com:80/")
    d = kil.canonical_url_hash("http://example.com/")
    assert c == d


def test_canonical_url_hash_different_urls():
    a = kil.canonical_url_hash("https://a.com")
    b = kil.canonical_url_hash("https://b.com")
    assert a != b


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def test_build_intake_link_body_includes_all_parts():
    body = kil.build_intake_link_body(
        url="https://example.com/article",
        context="Important read",
        note="Check sources",
        source="cli",
        board="attention-intake",
        assignee="link-analyst",
        idempotency_key="abc123",
        workspace_path="/tmp/fake-artifacts/t_xyz",
    )
    assert "https://example.com/article" in body
    assert "Important read" in body
    assert "Check sources" in body
    assert "needs_assessment" in body
    assert "attention-intake" in body
    assert "link-analyst" in body
    assert "abc123" in body
    assert "/tmp/fake-artifacts/t_xyz" in body
    assert "register.jsonl" in body


def test_build_intake_link_body_defaults():
    body = kil.build_intake_link_body(
        url="https://x.com",
        context=None,
        note=None,
        source="cli",
        board="attention-intake",
        assignee="link-analyst",
        idempotency_key="abc123",
        workspace_path="/tmp/fake-artifacts/t_xyz",
    )
    assert "(none provided)" in body
    assert "(none)" in body


# ---------------------------------------------------------------------------
# Title helper
# ---------------------------------------------------------------------------


def test_make_title_truncation():
    long_url = "https://example.com/" + "x" * 300
    title = kil._make_title(long_url)
    assert len(title) <= 140 + len("Link drop: ")
    assert title.startswith("Link drop:")


def test_make_title_short_url():
    title = kil._make_title("https://example.com")
    assert "example.com" in title


# ---------------------------------------------------------------------------
# create_intake_link — fresh
# ---------------------------------------------------------------------------


def test_create_intake_link_basic(conn):
    tid = kil.create_intake_link(
        conn,
        url="https://example.com/foo",
        context="ctx",
        note="nt",
    )
    assert tid.startswith("t_")
    task = kb.get_task(conn, tid)
    assert task.title.startswith("Link drop:")
    assert "https://example.com/foo" in task.body
    assert "needs_assessment" in task.body
    assert task.workspace_path is not None
    assert Path(task.workspace_path).exists()
    assert task.assignee == "link-analyst"
    assert task.status == "triage"
    assert task.idempotency_key == kil.canonical_url_hash("https://example.com/foo")


def test_create_intake_link_idempotency(conn):
    tid1 = kil.create_intake_link(conn, url="https://example.com/foo")
    tid2 = kil.create_intake_link(conn, url="https://example.com/foo")
    assert tid1 == tid2


def test_create_intake_link_idempotency_does_not_rewrite_completed_assessment(conn, tmp_path, monkeypatch):
    """A duplicate drop must not restore the provisional body over a completed assessment."""
    root = tmp_path / "attention-intake"
    monkeypatch.setattr(kil, "_artifact_root", lambda: root)

    tid = kil.create_intake_link(conn, url="https://example.com/done")
    completed_body = "Assess the link https://example.com/done and write final register entries."
    kb.update_task_body(conn, tid, completed_body)
    conn.execute("UPDATE tasks SET status = 'done', completed_at = 123456 WHERE id = ?", (tid,))
    conn.commit()

    before_rows = (root / "register.jsonl").read_text().splitlines()
    duplicate_tid = kil.create_intake_link(
        conn,
        url="https://example.com/done",
        context="new context should not overwrite assessment",
        note="duplicate drop",
    )

    final_task = kb.get_task(conn, tid)

    assert duplicate_tid == tid
    assert final_task is not None
    assert final_task.body == completed_body
    assert (root / "register.jsonl").read_text().splitlines() == before_rows


def test_create_intake_link_override_idempotency_key(conn):
    tid1 = kil.create_intake_link(
        conn,
        url="https://example.com/foo",
        idempotency_key="manual-key",
    )
    tid2 = kil.create_intake_link(
        conn,
        url="https://example.com/foo",
        idempotency_key="manual-key",
    )
    assert tid1 == tid2
    task = kb.get_task(conn, tid1)
    assert task.idempotency_key == "manual-key"


def test_create_intake_link_different_urls_different_ids(conn):
    tid1 = kil.create_intake_link(conn, url="https://a.com")
    tid2 = kil.create_intake_link(conn, url="https://b.com")
    assert tid1 != tid2


def test_create_intake_link_no_url_raises():
    with pytest.raises(ValueError, match="url is required"):
        kil.create_intake_link(None, url="")


# ---------------------------------------------------------------------------
# Register provisional write
# ---------------------------------------------------------------------------


def test_create_intake_link_writes_register_jsonl(conn, tmp_path, monkeypatch):
    # Override artifact root to tmp_path so we can inspect without mutating real state.
    root = tmp_path / "attention-intake"
    monkeypatch.setattr(kil, "_artifact_root", lambda: root)
    tid = kil.create_intake_link(conn, url="https://example.com/reg")
    jsonl = root / "register.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry["event"] == "intake_link_created"
    assert entry["task_id"] == tid
    assert entry["url"] == "https://example.com/reg"
    assert entry["status"] == "needs_assessment"


# ---------------------------------------------------------------------------
# Board override
# ---------------------------------------------------------------------------


def test_create_intake_link_board_override(conn):
    tid = kil.create_intake_link(
        conn,
        url="https://example.com/other",
        board="default",
    )
    task = kb.get_task(conn, tid)
    # board is stored only in the body template; helper itself doesn't
    # override kanban_db.create_task board arg (which defaults to None).
    # The board arg here is for the body text only.
    assert "default" in task.body


# ---------------------------------------------------------------------------
# Regression: default_workdir board auto-fill must not produce body-empty rows
# ---------------------------------------------------------------------------


def test_create_intake_link_with_board_default_workdir(conn, tmp_path, monkeypatch):
    """When board has a default_workdir, kanban_db fills workspace_path but
    create_intake_link must still patch body and register."""
    # Create a board with a default_workdir
    kb.create_board("attention-intake", default_workdir=str(tmp_path / "workdir"))
    tid = kil.create_intake_link(conn, url="https://example.com/dw")
    task = kb.get_task(conn, tid)
    assert task.body is not None
    assert "https://example.com/dw" in task.body
    assert "needs_assessment" in task.body
    assert task.workspace_path is not None


# ---------------------------------------------------------------------------
# Regression: write-path tests must not mutate the active register
# ---------------------------------------------------------------------------


def test_create_intake_link_does_not_mutate_active_register_jsonl(conn):
    """Regression probe: with HERMES_HOME isolation the real active
    register.jsonl must not be touched by any write path."""
    real_home = Path(os.environ.get("HOME", "/home/openclaw"))
    real_register = real_home / ".hermes" / "artifacts" / "attention-intake" / "register.jsonl"

    pre_mtime = real_register.stat().st_mtime if real_register.exists() else None
    pre_size = real_register.stat().st_size if real_register.exists() else None

    tid = kil.create_intake_link(conn, url="https://example.com/regression")
    assert tid.startswith("t_")

    if real_register.exists():
        post_stat = real_register.stat()
        assert post_stat.st_mtime == pre_mtime
        assert post_stat.st_size == pre_size
    else:
        assert not real_register.exists()

    # Sanity: the isolated tree DID get the write.
    isolated_root = Path(os.environ["HERMES_HOME"]) / "artifacts" / "attention-intake"
    assert (isolated_root / "register.jsonl").exists()
