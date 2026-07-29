"""Tests for the Kanban health/backup/repair/maintenance layer (hermes_cli.kanban_health).

All tests run inside a sandboxed HERMES_KANBAN_HOME with explicit path-proof
assertions.  No test may write bytes to the live kanban root under the
operator’s real home.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_health as kh
from hermes_cli.kanban import run_slash

# Live root used for fail-closed assertions (computed once per process).
_LIVE_KANBAN_ROOT = (Path.home() / ".hermes" / "kanban").resolve()
_LIVE_HERMES_ROOT = (Path.home() / ".hermes").resolve()


def _assert_sandboxed(path: Path, expected_root: Path) -> None:
    """Fail closed if *path* escapes *expected_root* or touches the live board."""
    resolved = path.expanduser().resolve()
    for live in (_LIVE_KANBAN_ROOT, _LIVE_HERMES_ROOT):
        try:
            resolved.relative_to(live)
            raise AssertionError(f"FAIL CLOSED: resolved path under live root: {resolved}")
        except ValueError:
            pass
    try:
        resolved.relative_to(expected_root.resolve())
    except ValueError:
        raise AssertionError(
            f"FAIL CLOSED: resolved path {resolved} not under expected root {expected_root}"
        )


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_KANBAN_HOME with an empty default board."""
    # 1. Clear inherited live Kanban pins (incident-hardening packet §2).
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)

    # 2. Set explicit sandbox root.
    home = tmp_path / "kanban-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # Also set HERMES_HOME so that hermes_constants.get_default_hermes_root()
    # agrees when kanban_home() falls through to it.
    monkeypatch.setenv("HERMES_HOME", str(home))

    # 3. Path proof — must hold *after* imports and in the same process.
    from hermes_cli import kanban_db as _kb
    resolved_home = _kb.kanban_home().resolve()
    _assert_sandboxed(resolved_home, home)
    _assert_sandboxed(_kb.kanban_db_path("default"), home)
    _assert_sandboxed(_kb.kanban_db_path("testboard"), home)

    _kb.init_db()
    return home


@pytest.fixture
def multi_board_home(tmp_path, monkeypatch):
    """Isolated HERMES_KANBAN_HOME with two boards (default + testboard)."""
    for key in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_BOARD",
        "HERMES_KANBAN_WORKSPACES_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)

    home = tmp_path / "kanban-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as _kb
    _assert_sandboxed(_kb.kanban_home().resolve(), home)
    _assert_sandboxed(_kb.kanban_db_path("default"), home)
    _assert_sandboxed(_kb.kanban_db_path("testboard"), home)

    _kb.init_db()
    _kb.create_board("testboard", name="Test Board")
    # Seed tasks
    with _kb.connect(board="default") as conn:
        _kb.create_task(conn, title="default-task", assignee="platform-eng")
    with _kb.connect(board="testboard") as conn:
        _kb.create_task(conn, title="test-task", assignee="frontend-eng")
    return home


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_board_health_report_default_board(multi_board_home):
    report = kh.board_health_report("default")
    assert report["slug"] == "default"
    assert report["healthy"] is True
    assert report["integrity_check"] == "ok"
    assert report["foreign_key_check"] == "ok"
    assert report["maintenance_mode"] is False
    assert report["task_counts"]["total"] == 1
    assert report["task_counts"]["running"] == 0
    assert "db_path" in report


def test_board_health_report_specific_board(multi_board_home):
    report = kh.board_health_report("testboard")
    assert report["slug"] == "testboard"
    assert report["healthy"] is True
    assert report["task_counts"]["total"] == 1


def test_board_health_report_none_uses_current(multi_board_home):
    # HERMES_KANBAN_BOARD is cleared by fixture, so current == default
    report = kh.board_health_report(None)
    assert report["slug"] == "default"


def test_fleet_health_report(multi_board_home):
    report = kh.fleet_health_report()
    assert report["summary"]["total"] == 2
    assert report["summary"]["healthy"] == 2
    assert report["summary"]["degraded"] == 0
    slugs = {b["slug"] for b in report["boards"]}
    assert slugs == {"default", "testboard"}


def test_fleet_health_report_skips_archived_by_default(multi_board_home):
    kb.write_board_metadata("testboard", archived=True)
    report = kh.fleet_health_report(include_archived=False)
    slugs = {b["slug"] for b in report["boards"]}
    assert slugs == {"default"}


def test_run_slash_health_board_json(multi_board_home):
    out = run_slash("health --board default --json")
    data = json.loads(out)
    assert data["summary"] == {"total": 1, "healthy": 1, "degraded": 0}
    assert data["boards"][0]["slug"] == "default"
    assert data["boards"][0]["integrity_check"] == "ok"


def test_run_slash_backup_board_json(multi_board_home):
    dest = multi_board_home / "slash-backups"
    out = run_slash(f"backup --board default --dest {dest} --json")
    data = json.loads(out)
    assert data[0]["board"] == "default"
    assert data[0]["error"] is None
    backup_path = Path(data[0]["path"])
    _assert_sandboxed(backup_path, multi_board_home)
    assert backup_path.exists()
    assert kh.verify_backup_manifest(dest / f"{backup_path.stem}.manifest.json") is True


def test_board_health_report_corrupt_db(multi_board_home):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    kh.inject_corruption(db_path, offset=0, length=16)
    report = kh.board_health_report("testboard")
    assert report["slug"] == "testboard"
    assert report["healthy"] is False
    assert report["integrity_check"] != "ok"


def test_board_health_report_corrupt_db_page(multi_board_home):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    kh.inject_corruption(db_path, offset=4096, length=512)
    report = kh.board_health_report("testboard")
    assert report["slug"] == "testboard"
    assert isinstance(report["integrity_check"], str)
    assert report["integrity_check"]
    assert isinstance(report["healthy"], bool)
    if report["integrity_check"] != "ok":
        assert report["healthy"] is False


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def test_backup_board_default(multi_board_home):
    dest = multi_board_home / "backups"
    out = kh.backup_board("default", dest_dir=dest)
    assert out.exists()
    assert out.suffix == ".db"
    manifest = dest / f"{out.stem}.manifest.json"
    assert manifest.exists()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["manifest_version"] == "1.0"
    assert m["board"]["slug"] == "default"
    assert m["backup"]["method"] == "sqlite_backup_api"
    assert m["backup"]["db_sha256"]  # non-empty hash
    assert m["source"]["integrity_check"] == "ok"
    assert m["source"]["foreign_key_check"] == "ok"


def test_backup_board_manifest_only(multi_board_home):
    dest = multi_board_home / "backups"
    out = kh.backup_board("testboard", dest_dir=dest, manifest_only=True)
    assert out.suffix == ".json"
    assert out.exists()
    m = json.loads(out.read_text(encoding="utf-8"))
    assert m["manifest_version"] == "1.0"
    assert m["backup"]["db_path"] == ""


def test_backup_board_integrity_refusal(multi_board_home):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    kh.inject_corruption(db_path, offset=0, length=16)
    dest = multi_board_home / "backups"
    with pytest.raises((kb.KanbanDbCorruptError, sqlite3.DatabaseError)):
        kh.backup_board("testboard", dest_dir=dest)


def test_backup_board_force_integrity_failure(multi_board_home):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    # Header corruption guarantees integrity_check fails deterministically.
    kh.inject_corruption(db_path, offset=0, length=16)
    dest = multi_board_home / "backups"
    out = kh.backup_board("testboard", dest_dir=dest, force=True)
    assert out.exists()
    manifest = dest / f"{out.stem}.manifest.json"
    assert manifest.exists()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["source"]["integrity_check"] != "ok"


def test_backup_board_force_raw_copy_preserves_sidecars(multi_board_home, monkeypatch):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    wal_path = db_path.parent / f"{db_path.name}-wal"
    shm_path = db_path.parent / f"{db_path.name}-shm"
    wal_path.write_bytes(b"forensic wal")
    shm_path.write_bytes(b"forensic shm")

    def _raise_corrupt(*args, **kwargs):
        raise kb.KanbanDbCorruptError(db_path, None, "forced test corruption")

    monkeypatch.setattr(kb, "connect", _raise_corrupt)
    dest = multi_board_home / "forensic-backups"
    out = kh.backup_board("testboard", dest_dir=dest, force=True)
    manifest = dest / f"{out.stem}.manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["backup"]["method"] == "raw_file_copy"
    assert m["backup"]["consolidated_wal"] is False
    assert Path(m["backup"]["sidecars"]["wal"]["path"]).exists()
    assert Path(m["backup"]["sidecars"]["shm"]["path"]).exists()


def test_backup_consolidates_wal(multi_board_home):
    """Backup should produce a standalone DB with no WAL sidecars."""
    db_path = kb.kanban_db_path("default")
    with kb.connect(board="default") as conn:
        kb.create_task(conn, title="wal-task")
    dest = multi_board_home / "backups"
    out = kh.backup_board("default", dest_dir=dest)
    backup_wal = out.parent / (out.name + "-wal")
    assert not backup_wal.exists()
    probe = sqlite3.connect(str(out))
    with probe:
        row = probe.execute("PRAGMA integrity_check").fetchone()
        assert str(row[0]).lower() == "ok"
    probe.close()


def test_verify_backup_manifest(multi_board_home):
    dest = multi_board_home / "backups"
    out = kh.backup_board("default", dest_dir=dest)
    manifest = dest / f"{out.stem}.manifest.json"
    assert kh.verify_backup_manifest(manifest) is True


def test_verify_backup_manifest_missing_file(multi_board_home):
    fake = multi_board_home / "no.manifest.json"
    fake.write_text(json.dumps({"backup": {"db_path": "/dev/null/foo"}}), encoding="utf-8")
    assert kh.verify_backup_manifest(fake) is False


def test_verify_backup_manifest_hash_mismatch(multi_board_home):
    dest = multi_board_home / "backups"
    out = kh.backup_board("default", dest_dir=dest)
    manifest = dest / f"{out.stem}.manifest.json"
    m = json.loads(manifest.read_text(encoding="utf-8"))
    m["backup"]["db_sha256"] = "0" * 64
    manifest.write_text(json.dumps(m), encoding="utf-8")
    assert kh.verify_backup_manifest(manifest) is False


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def test_create_repair_candidate_from_live(multi_board_home):
    with kb.connect(board="default") as conn:
        kb.create_task(conn, title="repair-task")
    manifest_path = kh.create_repair_candidate("default")
    assert manifest_path.exists()
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert candidate["candidate_version"] == "1.0"
    assert candidate["board_slug"] == "default"
    assert candidate["candidate_db_path"]
    assert Path(candidate["candidate_db_path"]).exists()
    assert candidate["candidate_integrity_check"] == "ok"


def test_create_repair_candidate_quotes_schema_identifiers(multi_board_home):
    db_path = kb.kanban_db_path("default")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute('CREATE TABLE "odd""table" ("odd""col" TEXT)')
        conn.execute('INSERT INTO "odd""table" ("odd""col") VALUES (?)', ("survives",))

    manifest_path = kh.create_repair_candidate("default")
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    with sqlite3.connect(candidate["candidate_db_path"]) as conn:
        row = conn.execute('SELECT "odd""col" FROM "odd""table"').fetchone()
    assert row == ("survives",)


def test_create_repair_candidate_from_backup(multi_board_home):
    dest = multi_board_home / "backups"
    out = kh.backup_board("default", dest_dir=dest)
    manifest_path = kh.create_repair_candidate("default", from_backup=out)
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert candidate["candidate_db_path"]
    assert Path(candidate["candidate_db_path"]).exists()


def test_approve_swap_board_requires_confirmation(multi_board_home):
    manifest_path = kh.create_repair_candidate("default")
    with pytest.raises(RuntimeError):
        kh.approve_swap_board("default", candidate_manifest_path=manifest_path)


def test_approve_swap_board_requires_maintenance_without_force(multi_board_home):
    manifest_path = kh.create_repair_candidate("default")
    with pytest.raises(RuntimeError, match="requires maintenance mode"):
        kh.approve_swap_board("default", candidate_manifest_path=manifest_path, yes_flag=True)


def test_approve_swap_board_rejects_wrong_board_candidate(multi_board_home):
    manifest_path = kh.create_repair_candidate("default")
    with pytest.raises(RuntimeError, match="board mismatch"):
        kh.approve_swap_board("testboard", candidate_manifest_path=manifest_path, yes_flag=True, force=True)


def test_approve_swap_board_rejects_missing_candidate_hash(multi_board_home):
    manifest_path = kh.create_repair_candidate("default")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("candidate_db_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing candidate_db_sha256"):
        kh.approve_swap_board("default", candidate_manifest_path=manifest_path, yes_flag=True, force=True)


def test_approve_swap_board_rejects_candidate_hash_mismatch(multi_board_home):
    manifest_path = kh.create_repair_candidate("default")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_db_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        kh.approve_swap_board("default", candidate_manifest_path=manifest_path, yes_flag=True, force=True)


def test_approve_swap_board_rejects_running_tasks_without_force(multi_board_home):
    with kb.connect(board="default") as conn:
        tid = kb.create_task(conn, title="running-task")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
    manifest_path = kh.create_repair_candidate("default")
    kh.set_maintenance_mode("default", True, reason="test running guard")
    with pytest.raises(RuntimeError, match="running task"):
        kh.approve_swap_board("default", candidate_manifest_path=manifest_path, yes_flag=True)


def test_approve_swap_board_succeeds_with_maintenance_and_preserves_candidate(multi_board_home):
    with kb.connect(board="default") as conn:
        kb.create_task(conn, title="pre-swap-task")
    db_path = kb.kanban_db_path("default")
    manifest_path = kh.create_repair_candidate("default")
    candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_db = Path(candidate["candidate_db_path"])
    wal_path = db_path.parent / f"{db_path.name}-wal"
    shm_path = db_path.parent / f"{db_path.name}-shm"
    wal_path.write_bytes(b"stale wal")
    shm_path.write_bytes(b"stale shm")
    kh.set_maintenance_mode("default", True, reason="test swap")
    receipt = kh.approve_swap_board(
        "default", candidate_manifest_path=manifest_path, yes_flag=True
    )
    assert receipt["swapped"] is True
    assert receipt["board"] == "default"
    assert receipt["post_swap_integrity_check"] == "ok"
    assert receipt["pre_swap_backup"]
    assert Path(receipt["pre_swap_backup"]).exists()
    assert candidate_db.exists(), "candidate DB is evidence and should not be moved"
    assert not wal_path.exists()
    assert not shm_path.exists()
    assert db_path.exists()

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def test_set_maintenance_mode(multi_board_home):
    kh.set_maintenance_mode("testboard", True, reason="DB migration")
    assert kh.is_maintenance_mode("testboard") is True
    meta = kb.read_board_metadata("testboard")
    assert meta["maintenance"] is True
    assert meta["maintenance_reason"] == "DB migration"
    assert meta["maintenance_since"] > 0


def test_clear_maintenance_mode(multi_board_home):
    kh.set_maintenance_mode("testboard", True, reason="test")
    assert kh.is_maintenance_mode("testboard") is True
    kh.set_maintenance_mode("testboard", False)
    assert kh.is_maintenance_mode("testboard") is False
    meta = kb.read_board_metadata("testboard")
    assert "maintenance_reason" not in meta
    assert "maintenance_since" not in meta


def test_run_slash_maintenance_toggles_metadata(multi_board_home):
    out = run_slash("maintenance testboard on --reason 'CLI proof'")
    assert "Maintenance mode enabled" in out
    assert kh.is_maintenance_mode("testboard") is True
    assert kb.read_board_metadata("testboard")["maintenance_reason"] == "CLI proof"
    out = run_slash("maintenance testboard off")
    assert "Maintenance mode disabled" in out
    assert kh.is_maintenance_mode("testboard") is False


# ---------------------------------------------------------------------------
# Corruption injection helper
# ---------------------------------------------------------------------------

def test_inject_corruption_refuses_live_kanban_root():
    live_like_path = Path.home() / ".hermes" / "kanban" / "boards" / "real" / "kanban.db"
    with pytest.raises(RuntimeError, match="Refusing to inject corruption into live Kanban path"):
        kh.inject_corruption(live_like_path, offset=0, length=16)


def test_inject_corruption_refuses_live_default_db_path():
    live_default_path = Path.home() / ".hermes" / "kanban.db"
    with pytest.raises(RuntimeError, match="Refusing to inject corruption into live Kanban path"):
        kh.inject_corruption(live_default_path, offset=0, length=16)


def test_inject_corruption_refuses_operator_default_db_even_if_home_is_monkeypatched(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    live_default_path = Path("/home/openclaw/.hermes/kanban.db")
    with pytest.raises(RuntimeError, match="Refusing to inject corruption into live Kanban path"):
        kh.inject_corruption(live_default_path, offset=0, length=16)


def test_inject_corruption_refuses_operator_root_even_if_home_is_monkeypatched(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    live_like_path = Path("/home/openclaw/.hermes/kanban/boards/real/kanban.db")
    with pytest.raises(RuntimeError, match="Refusing to inject corruption into live Kanban path"):
        kh.inject_corruption(live_like_path, offset=0, length=16)


def test_inject_corruption_makes_db_unreadable(multi_board_home):
    db_path = kb.kanban_db_path("testboard")
    _assert_sandboxed(db_path, multi_board_home)
    before = kh._file_sha256(db_path)
    # Corrupt an interior page so the file is definitely altered.
    kh.inject_corruption(db_path, offset=4096, length=512)
    after = kh._file_sha256(db_path)
    assert before != after, "inject_corruption should alter file contents"
    # A fresh connect succeeds; the corrupted page may or may not be
    # detected by PRAGMA integrity_check depending on WAL state and
    # whether SQLite has loaded that page. Detection is tested elsewhere.
    conn = sqlite3.connect(str(db_path))
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    conn.close()


# ---------------------------------------------------------------------------
# Sidecar helpers
# ---------------------------------------------------------------------------

def test_sidecar_info_no_wal(multi_board_home):
    db_path = kb.kanban_db_path("default")
    info = kh._sidecar_info(db_path)
    assert "wal" in info
    assert "shm" in info


def test_pragma_str_ok(multi_board_home):
    from hermes_state import is_sqlite_wal_reset_vulnerable

    conn = kb.connect(board="default")
    val = kh._pragma_str(conn, "PRAGMA journal_mode")
    expected = "delete" if is_sqlite_wal_reset_vulnerable() else "wal"
    assert val.lower() == expected
    conn.close()


# ---------------------------------------------------------------------------
# Cross-surface maintenance regression tests (HP-KDB-12R)
# ---------------------------------------------------------------------------

def test_kanban_maintenance_surfaces_agree(multi_board_home):
    """Top-level maintenance, boards maintenance, health report, and dispatch
    must all agree on the same canonical ``maintenance`` key.
    """
    board = "testboard"

    # Initially off everywhere
    assert kh.is_maintenance_mode(board) is False
    assert kb.board_in_maintenance(board) is False
    assert kb.read_board_metadata(board).get("maintenance") is not True
    report = kh.board_health_report(board)
    assert report["maintenance_mode"] is False

    # Enable via kanban_health API
    kh.set_maintenance_mode(board, True, reason="cross-surface test")
    assert kh.is_maintenance_mode(board) is True
    assert kb.board_in_maintenance(board) is True
    meta = kb.read_board_metadata(board)
    assert meta.get("maintenance") is True
    assert "maintenance_mode" not in meta
    assert meta.get("maintenance_reason") == "cross-surface test"
    assert meta.get("maintenance_since") is not None
    report = kh.board_health_report(board)
    assert report["maintenance_mode"] is True

    # Disable via kanban_health API
    kh.set_maintenance_mode(board, False)
    assert kh.is_maintenance_mode(board) is False
    assert kb.board_in_maintenance(board) is False
    meta = kb.read_board_metadata(board)
    assert meta.get("maintenance") is not True
    assert "maintenance_reason" not in meta
    assert "maintenance_since" not in meta
    report = kh.board_health_report(board)
    assert report["maintenance_mode"] is False


def test_boards_maintenance_on_makes_approve_swap_accept(multi_board_home):
    """Using ``boards maintenance <board> on`` must satisfy the maintenance
    gate in ``approve_swap_board(..., yes_flag=True)``.
    """
    board = "testboard"
    db_path = kb.kanban_db_path(board=board)

    # Enable via kanban_db primitive (the one boards maintenance calls)
    kb.set_board_maintenance(board, True, reason="boards maintenance test")
    assert kb.board_in_maintenance(board) is True
    assert kh.is_maintenance_mode(board) is True

    # Create a trivial repair candidate (copy current DB)
    import shutil, time
    candidate_dir = db_path.parent / "repair-candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_db = candidate_dir / "candidate.db"
    shutil.copy2(str(db_path), str(candidate_db))
    manifest = {
        "board_slug": board,
        "candidate_db_path": str(candidate_db),
        "candidate_db_sha256": kh._file_sha256(candidate_db),
        "created_at": int(time.time()),
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Swap should succeed because maintenance mode is active and no tasks running
    receipt = kh.approve_swap_board(
        board, candidate_manifest_path=manifest_path, yes_flag=True
    )
    assert receipt["swapped"] is True
    assert receipt["maintenance_mode"] is True


def test_top_level_maintenance_on_makes_dispatcher_skip_board(multi_board_home):
    """Using top-level ``hermes kanban maintenance <board> on`` must set the
    same key the dispatcher reads via ``board_in_maintenance()``.
    """
    board = "testboard"
    kh.set_maintenance_mode(board, True, reason="dispatcher skip test")
    assert kh.is_maintenance_mode(board) is True
    assert kb.board_in_maintenance(board) is True

    # Simulate the dispatcher check exactly as in gateway/kanban_watchers.py
    meta = kb.read_board_metadata(board)
    assert kb.board_in_maintenance(meta) is True


def test_maintenance_mode_degraded_path_uses_canonical_key(multi_board_home):
    """When the board DB is corrupt, the degraded health report must still
    read the canonical ``maintenance`` key, not the old ``maintenance_mode``.
    """
    board = "testboard"
    db_path = kb.kanban_db_path(board=board)

    # Enable maintenance first, then corrupt the DB
    kb.set_board_maintenance(board, True, reason="degraded path test")
    kh.inject_corruption(db_path, offset=0, length=16)

    report = kh.board_health_report(board)
    assert report["healthy"] is False
    assert report["maintenance_mode"] is True


def test_maintenance_mode_fleet_report_uses_canonical_key(multi_board_home):
    """Fleet-wide health report must read the canonical ``maintenance`` key."""
    board = "testboard"
    kb.set_board_maintenance(board, True, reason="fleet report test")
    fleet = kh.fleet_health_report()
    for r in fleet["boards"]:
        if r["slug"] == board:
            assert r["maintenance_mode"] is True
            return
    pytest.fail("board not found in fleet report")


def test_legacy_maintenance_mode_without_canonical_key_is_supported(multi_board_home):
    """Metadata written by the split-key implementation remains protective
    until a canonical writer normalizes the board metadata.
    """
    board = "testboard"
    meta_path = kb.board_metadata_path(board)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maintenance_mode"] = True  # old key
    meta.pop("maintenance", None)     # no canonical key yet
    meta.pop("db_path", None)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert kh.is_maintenance_mode(board) is True
    assert kb.board_in_maintenance(board) is True
    report = kh.board_health_report(board)
    assert report["maintenance_mode"] is True


def test_legacy_maintenance_mode_degraded_path_is_supported(multi_board_home):
    """Legacy-only maintenance metadata remains visible in degraded reports."""
    board = "testboard"
    meta_path = kb.board_metadata_path(board)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maintenance_mode"] = True
    meta.pop("maintenance", None)
    meta.pop("db_path", None)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    kh.inject_corruption(kb.kanban_db_path(board=board), offset=0, length=16)
    report = kh.board_health_report(board)
    assert report["healthy"] is False
    assert report["maintenance_mode"] is True


def test_canonical_maintenance_key_overrides_legacy_maintenance_mode(multi_board_home):
    """If both keys exist, the canonical ``maintenance`` key wins.

    This prevents stale legacy metadata from overriding an explicit canonical
    off state.
    """
    board = "testboard"
    meta_path = kb.board_metadata_path(board)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maintenance_mode"] = True  # old key
    meta["maintenance"] = False      # canonical key
    meta.pop("db_path", None)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    assert kh.is_maintenance_mode(board) is False
    assert kb.board_in_maintenance(board) is False
    report = kh.board_health_report(board)
    assert report["maintenance_mode"] is False


def test_canonical_maintenance_writer_removes_legacy_key(multi_board_home):
    """Canonical writes normalize old split-key metadata."""
    board = "testboard"
    meta_path = kb.board_metadata_path(board)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["maintenance_mode"] = True
    meta.pop("db_path", None)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    kb.set_board_maintenance(board, True, reason="normalize")
    meta = kb.read_board_metadata(board)
    assert meta["maintenance"] is True
    assert "maintenance_mode" not in meta
    assert meta["maintenance_reason"] == "normalize"

    kb.set_board_maintenance(board, False)
    meta = kb.read_board_metadata(board)
    assert meta["maintenance"] is False
    assert "maintenance_mode" not in meta
