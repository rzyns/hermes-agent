"""Kanban health, backup, repair, and maintenance primitives.

Provides SQLite-aware backup (using sqlite3.Connection.backup), fleet-wide
health reports, repair-candidate creation, approval-gated board swap, and
maintenance-mode toggling. All functions are safe against WAL-mode boards and
use temp-HERMES_HOME fixture patterns for testability.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _quote_sqlite_identifier(identifier: str) -> str:
    """Quote a SQLite identifier sourced from schema metadata."""
    if "\x00" in identifier:
        raise ValueError("SQLite identifiers cannot contain NUL bytes")
    return '"' + identifier.replace('"', '""') + '"'


def _file_sha256(path: Path) -> str:
    """Return a hex SHA-256 digest for a file."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _sidecar_info(db_path: Path) -> dict[str, Any]:
    """Return WAL/SHM presence, size, and mtime for a DB path."""
    parent = db_path.parent
    base = db_path.name
    info: dict[str, Any] = {"wal": None, "shm": None}
    for suffix in ("-wal", "-shm"):
        sidecar = parent / (base + suffix)
        key = suffix.lstrip("-")
        if sidecar.exists() and sidecar.is_file():
            stat = sidecar.stat()
            info[key] = {
                "present": True,
                "size_bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
        else:
            info[key] = {"present": False}
    return info


def _pragma_str(conn: sqlite3.Connection, pragma: str) -> str:
    """Run a PRAGMA that returns a single text column, return it safely."""
    try:
        row = conn.execute(pragma).fetchone()
        if row is not None:
            val = row[0]
            return str(val) if val is not None else ""
    except Exception:
        pass
    return ""


def _integrity_check(conn: sqlite3.Connection) -> tuple[bool, str]:
    """Run PRAGMA integrity_check; return (ok, message_or_detail)."""
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if row and str(row[0]).lower() == "ok":
            return True, "ok"
        return False, str(row[0]) if row else "<no row>"
    except sqlite3.OperationalError:
        # Lock/busy — re-raise as transient, not corruption.
        raise
    except sqlite3.DatabaseError as exc:
        return False, f"sqlite refused: {exc}"


def _foreign_key_check(conn: sqlite3.Connection) -> tuple[bool, str]:
    """Run PRAGMA foreign_key_check; return (ok, message_or_detail)."""
    try:
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        if not rows:
            return True, "ok"
        return False, f"{len(rows)} violation(s)"
    except sqlite3.OperationalError:
        raise
    except sqlite3.DatabaseError as exc:
        return False, f"sqlite refused: {exc}"


def _sqlite_version() -> str:
    return sqlite3.sqlite_version


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def board_health_report(
    board: Optional[str] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, Any]:
    """Return a health report dict for a single board (read-only on source DB).

    If the board DB is corrupt, returns a degraded report instead of raising.
    Matches the schema in HP-KDB-02 §4.2.
    """
    close = conn is None
    if conn is None:
        try:
            conn = kb.connect(board=board)
        except kb.KanbanDbCorruptError as exc:
            db_path = kb.kanban_db_path(board=board)
            sidecars = _sidecar_info(db_path)
            try:
                meta = kb.read_board_metadata(board)
                slug = meta.get("slug", board or kb.DEFAULT_BOARD)
                maintenance = kb.board_in_maintenance(meta)
            except Exception:
                slug = board or kb.DEFAULT_BOARD
                maintenance = False
            return {
                "slug": slug,
                "db_path": str(db_path),
                "healthy": False,
                "integrity_check": str(exc.reason) if hasattr(exc, "reason") else str(exc),
                "foreign_key_check": "unknown (db refused to open)",
                "wal": sidecars["wal"],
                "shm": sidecars["shm"],
                "task_counts": {"total": 0, "running": 0, "stale": 0},
                "maintenance_mode": maintenance,
            }
        except sqlite3.DatabaseError as exc:
            db_path = kb.kanban_db_path(board=board)
            sidecars = _sidecar_info(db_path)
            try:
                meta = kb.read_board_metadata(board)
                slug = meta.get("slug", board or kb.DEFAULT_BOARD)
                maintenance = kb.board_in_maintenance(meta)
            except Exception:
                slug = board or kb.DEFAULT_BOARD
                maintenance = False
            return {
                "slug": slug,
                "db_path": str(db_path),
                "healthy": False,
                "integrity_check": str(exc),
                "foreign_key_check": "unknown (db refused to open)",
                "wal": sidecars["wal"],
                "shm": sidecars["shm"],
                "task_counts": {"total": 0, "running": 0, "stale": 0},
                "maintenance_mode": maintenance,
            }
    try:
        db_path = kb.kanban_db_path(board=board)
        meta = kb.read_board_metadata(board)
        slug = meta.get("slug", board or kb.DEFAULT_BOARD)

        # Integrity checks
        integrity_ok, integrity_msg = _integrity_check(conn)
        fk_ok, fk_msg = _foreign_key_check(conn)
        healthy = integrity_ok and fk_ok

        # WAL/SHM sidecars
        sidecars = _sidecar_info(db_path)

        # Task counts. A corrupt page can become visible only after the broad
        # integrity probes have returned (or only to this specific query),
        # especially when the WAL-reset safety guard selects DELETE mode.
        # Health reporting must degrade rather than raise on that late read.
        try:
            counts = kb.board_stats(conn)
        except sqlite3.DatabaseError as exc:
            counts = {"by_status": {}}
            healthy = False
            if integrity_ok:
                integrity_msg = str(exc)
        by_status = counts.get("by_status", {})
        total = sum(by_status.values())
        running = by_status.get("running", 0)

        # Maintenance mode
        maintenance = kb.board_in_maintenance(meta)

        report = {
            "slug": slug,
            "db_path": str(db_path),
            "healthy": healthy,
            "integrity_check": integrity_msg,
            "foreign_key_check": fk_msg,
            "wal": sidecars["wal"],
            "shm": sidecars["shm"],
            "task_counts": {
                "total": total,
                "running": running,
                "stale": 0,
            },
            "maintenance_mode": maintenance,
        }
        return report
    finally:
        if close and conn is not None:
            conn.close()


def fleet_health_report(
    *,
    include_archived: bool = False,
) -> dict[str, Any]:
    """Return health reports for all boards (read-only on each source DB)."""
    boards = kb.list_boards(include_archived=include_archived)
    reports: list[dict[str, Any]] = []
    healthy_count = 0
    degraded_count = 0
    for meta in boards:
        slug = meta["slug"]
        try:
            report = board_health_report(slug)
        except sqlite3.OperationalError:
            # Busy/locked — surface as degraded, not fatal.
            report = {
                "slug": slug,
                "db_path": meta.get("db_path", ""),
                "healthy": False,
                "integrity_check": "locked/busy",
                "foreign_key_check": "unknown",
                "wal": None,
                "shm": None,
                "task_counts": {"total": 0, "running": 0, "stale": 0},
                "maintenance_mode": kb.board_in_maintenance(meta),
            }
        if report["healthy"]:
            healthy_count += 1
        else:
            degraded_count += 1
        reports.append(report)
    return {
        "boards": reports,
        "summary": {
            "total": len(reports),
            "healthy": healthy_count,
            "degraded": degraded_count,
        },
    }


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_board(
    board: Optional[str] = None,
    *,
    dest_dir: Path,
    label: Optional[str] = None,
    manifest_only: bool = False,
    force: bool = False,
) -> Path:
    """SQLite-aware backup of a board DB with manifest generation.

    Uses sqlite3.Connection.backup() to produce a consistent standalone DB
    (WAL consolidated into the backup file). Raises KanbanDbCorruptError if
    the source fails integrity checks and force=False.

    Returns the path to the backup DB file.
    """
    db_path = kb.kanban_db_path(board=board)
    meta = kb.read_board_metadata(board)
    slug = meta.get("slug", board or kb.DEFAULT_BOARD)
    now_iso = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base_name = f"kanban-backup-{slug}-{now_iso}"
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    backup_db_path = dest_dir / f"{base_name}.db"
    manifest_path = dest_dir / f"{base_name}.manifest.json"

    # Pre-flight integrity on source
    try:
        preflight_conn = kb.connect(board=board)
    except (kb.KanbanDbCorruptError, sqlite3.DatabaseError) as exc:
        source_integrity_ok = False
        source_integrity_msg = str(exc)
        preflight_conn = None
    else:
        try:
            source_integrity_ok, source_integrity_msg = _integrity_check(preflight_conn)
        finally:
            preflight_conn.close()

    if not source_integrity_ok and not force:
        raise kb.KanbanDbCorruptError(
            db_path, None,
            f"backup refused: integrity_check failed ({source_integrity_msg})"
        )

    # Manifest source info (computed from live file BEFORE backup)
    source_sha = _file_sha256(db_path)
    source_size = db_path.stat().st_size if db_path.exists() else 0
    sidecars = _sidecar_info(db_path)
    wal_sha = _file_sha256(db_path.parent / (db_path.name + "-wal")) if sidecars["wal"] and sidecars["wal"].get("present") else ""

    with contextlib.ExitStack() as stack:
        backup_method = "sqlite_backup_api"
        consolidated_wal = True
        backup_sidecars: dict[str, dict[str, Any]] = {}
        try:
            src_conn = stack.enter_context(kb.connect_closing(board=board))
        except (kb.KanbanDbCorruptError, sqlite3.DatabaseError) as exc:
            if not force:
                raise
            # Forensic fallback: raw file + sidecar copy when SQLite cannot
            # open the DB. This is not a clean standalone backup, so the
            # manifest must not claim WAL consolidation.
            shutil.copy2(str(db_path), str(backup_db_path))
            backup_sha = _file_sha256(backup_db_path)
            backup_size = backup_db_path.stat().st_size
            fk_msg = f"unavailable — {exc}"
            journal_mode = "unknown"
            synchronous = "unknown"
            backup_method = "raw_file_copy"
            consolidated_wal = False
            for suffix, key in (("-wal", "wal"), ("-shm", "shm")):
                source_sidecar = db_path.parent / (db_path.name + suffix)
                if source_sidecar.exists() and source_sidecar.is_file():
                    dest_sidecar = backup_db_path.parent / (backup_db_path.name + suffix)
                    shutil.copy2(str(source_sidecar), str(dest_sidecar))
                    backup_sidecars[key] = {
                        "path": str(dest_sidecar),
                        "sha256": _file_sha256(dest_sidecar),
                        "size_bytes": dest_sidecar.stat().st_size,
                    }
        else:
            fk_ok, fk_msg = _foreign_key_check(src_conn)
            journal_mode = _pragma_str(src_conn, "PRAGMA journal_mode")
            synchronous = _pragma_str(src_conn, "PRAGMA synchronous")

            if not manifest_only:
                # SQLite backup API — produces a clean standalone DB (no WAL sidecars).
                # Use a temp file first, then move into place.
                tmp_fd, tmp_path_str = tempfile.mkstemp(
                    dir=str(dest_dir), prefix=f"{base_name}.tmp", suffix=".db"
                )
                os.close(tmp_fd)
                tmp_path = Path(tmp_path_str)
                stack.callback(lambda p=tmp_path: p.unlink(missing_ok=True))
                dst_conn = sqlite3.connect(str(tmp_path))
                stack.callback(dst_conn.close)
                with dst_conn:
                    src_conn.backup(dst_conn)
                # Move to final name atomically
                os.replace(str(tmp_path), str(backup_db_path))
                backup_sha = _file_sha256(backup_db_path)
                backup_size = backup_db_path.stat().st_size
            else:
                backup_sha = ""
                backup_size = 0

    # Build manifest
    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board": {
            "slug": slug,
            "display_name": meta.get("name", slug),
            "db_path": str(db_path),
        },
        "source": {
            "db_sha256": source_sha,
            "db_size_bytes": source_size,
            "wal_present": bool(sidecars["wal"] and sidecars["wal"].get("present")),
            "wal_sha256": wal_sha,
            "wal_size_bytes": sidecars["wal"].get("size_bytes", 0) if sidecars["wal"] else 0,
            "shm_present": bool(sidecars["shm"] and sidecars["shm"].get("present")),
            "shm_size_bytes": sidecars["shm"].get("size_bytes", 0) if sidecars["shm"] else 0,
            "integrity_check": source_integrity_msg,
            "foreign_key_check": fk_msg,
            "sqlite_version": _sqlite_version(),
            "journal_mode": journal_mode,
            "synchronous": synchronous,
        },
        "backup": {
            "db_path": str(backup_db_path) if not manifest_only else "",
            "db_sha256": backup_sha,
            "method": backup_method,
            "consolidated_wal": consolidated_wal,
            "sqlite_page_count": 0,  # populated below if file exists
            "sidecars": backup_sidecars,
        },
        "label": label or "",
        "hermes_version": "2.x.x",
    }

    if backup_method == "sqlite_backup_api" and backup_db_path.exists():
        try:
            probe = sqlite3.connect(str(backup_db_path))
            with probe:
                page_count = probe.execute("PRAGMA page_count").fetchone()[0]
            manifest["backup"]["sqlite_page_count"] = page_count
            probe.close()
        except Exception:
            pass

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return backup_db_path if not manifest_only else manifest_path


def verify_backup_manifest(
    manifest_path: Path,
    *,
    check_source_exists: bool = True,
) -> bool:
    """Verify a backup manifest against the files it references.

    Returns True if the manifest is internally consistent and the backup
    DB hash matches what the manifest recorded.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    backup_info = manifest.get("backup", {})
    backup_db_path = Path(backup_info.get("db_path", ""))
    if not backup_db_path.exists():
        return False
    expected = backup_info.get("db_sha256", "")
    if expected and _file_sha256(backup_db_path) != expected:
        return False

    if check_source_exists:
        source_info = manifest.get("source", {})
        src_path = Path(manifest.get("board", {}).get("db_path", ""))
        if src_path.exists() and source_info.get("db_sha256"):
            # Source may have changed; just check file exists.
            pass

    return True


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def create_repair_candidate(
    board: str,
    *,
    from_backup: Optional[Path] = None,
    reason: str = "",
) -> Path:
    """Create a repair candidate directory + manifest from a board.

    Best-effort: opens the current (potentially corrupt) board, dumps
    readable rows into a new candidate DB, and writes a candidate manifest.
    Does NOT touch the live board.

    Returns the candidate manifest path.
    """
    slug = kb._normalize_board_slug(board) or board
    db_path = kb.kanban_db_path(board=slug)
    now_iso = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    candidate_dir = Path(tempfile.mkdtemp(prefix=f"hermes-repair-{slug}-{now_iso}-"))
    candidate_db = candidate_dir / "candidate.db"
    manifest_path = candidate_dir / "repair-candidate.json"

    source_sha = _file_sha256(db_path)

    # Best-effort source integrity result
    try:
        src_conn = sqlite3.connect(str(db_path))
        with src_conn:
            row = src_conn.execute("PRAGMA integrity_check").fetchone()
            integrity_result = str(row[0]) if row else "<no row>"
    except Exception as exc:
        integrity_result = f"could not probe: {exc}"
    finally:
        try:
            src_conn.close()
        except Exception:
            pass

    # Build candidate DB by copying schema and readable rows
    recovered: dict[str, int] = {}
    if from_backup is not None and from_backup.exists():
        # If a clean backup is provided, use that as base
        shutil = __import__("shutil")
        shutil.copy2(str(from_backup), str(candidate_db))
        recovered = {"tasks": 0, "task_comments": 0, "task_events": 0,
                     "task_links": 0, "task_runs": 0}
    else:
        # Best-effort dump from corrupt source
        try:
            src_conn = sqlite3.connect(str(db_path))
            cursor = src_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]
            dst_conn = sqlite3.connect(str(candidate_db))
            for table in tables:
                try:
                    quoted_table = _quote_sqlite_identifier(table)
                except ValueError:
                    continue
                try:
                    # Get CREATE TABLE
                    cursor.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,))
                    create_row = cursor.fetchone()
                    if create_row and create_row[0]:
                        dst_conn.execute(create_row[0])
                except Exception:
                    pass
                # Copy rows
                row_count = 0
                try:
                    cursor.execute(f"SELECT * FROM {quoted_table}")
                    cols = [d[0] for d in cursor.description]
                    quoted_cols = ",".join(_quote_sqlite_identifier(col) for col in cols)
                    placeholders = ",".join("?" for _ in cols)
                    for row in cursor:
                        try:
                            dst_conn.execute(
                                f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES ({placeholders})",
                                row,
                            )
                            row_count += 1
                        except Exception:
                            pass
                except Exception:
                    pass
                recovered[table] = row_count
            dst_conn.commit()
            dst_conn.close()
            src_conn.close()
        except Exception:
            pass

    candidate_integrity = "unknown"
    try:
        probe = sqlite3.connect(str(candidate_db))
        with probe:
            row = probe.execute("PRAGMA integrity_check").fetchone()
            candidate_integrity = str(row[0]) if row else "<no row>"
        probe.close()
    except Exception:
        pass

    candidate_sha = _file_sha256(candidate_db)

    manifest: dict[str, Any] = {
        "candidate_version": "1.0",
        "board_slug": slug,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "original_db_path": str(db_path),
            "original_db_sha256": source_sha,
            "integrity_check_result": integrity_result,
        },
        "recovery_method": "sqlite_best_effort_dump",
        "recovered_rows": recovered,
        "candidate_db_path": str(candidate_db),
        "candidate_db_sha256": candidate_sha,
        "candidate_integrity_check": candidate_integrity,
        "missing_rows_estimate": 0,
        "notes": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _probe_db_integrity(db_path: Path) -> str:
    """Return PRAGMA integrity_check text for a standalone DB file."""
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "<no row>"
    except sqlite3.DatabaseError as exc:
        return f"sqlite refused: {exc}"


def _running_task_count(board: str) -> int:
    """Best-effort count of running tasks for active-use guardrails."""
    try:
        with kb.connect_closing(board=board) as conn:
            stats = kb.board_stats(conn)
        return int(stats.get("by_status", {}).get("running", 0))
    except Exception:
        # If the board is corrupt/unopenable, do not treat this probe as an
        # active-use proof. Maintenance-mode confirmation still gates swaps.
        return 0


def _sidecar_paths(db_path: Path) -> tuple[Path, Path]:
    return (
        db_path.parent / f"{db_path.name}-wal",
        db_path.parent / f"{db_path.name}-shm",
    )


def approve_swap_board(
    board: str,
    *,
    candidate_manifest_path: Path,
    force: bool = False,
    yes_flag: bool = False,
) -> dict[str, Any]:
    """Approval-gated live swap of a board DB with a repair candidate.

    Creates a SQLite-aware pre-swap backup of the current DB, validates that
    the candidate manifest belongs to the requested board, removes stale
    WAL/SHM sidecars, then atomically replaces the board DB with a copied
    candidate. The repair candidate itself is preserved as evidence.

    Returns a swap receipt dict.
    """
    slug = kb._normalize_board_slug(board) or board
    db_path = kb.kanban_db_path(board=slug)
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    manifest_board = kb._normalize_board_slug(candidate_manifest.get("board_slug"))
    if manifest_board != slug:
        raise RuntimeError(
            f"repair candidate board mismatch: manifest is {manifest_board!r}, requested {slug!r}"
        )

    candidate_db = Path(candidate_manifest["candidate_db_path"])
    if not candidate_db.exists():
        raise FileNotFoundError(f"candidate DB not found: {candidate_db}")

    expected_sha = str(candidate_manifest.get("candidate_db_sha256", ""))
    if not expected_sha:
        raise RuntimeError("repair candidate manifest missing candidate_db_sha256")
    actual_sha = _file_sha256(candidate_db)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"repair candidate hash mismatch: expected {expected_sha}, got {actual_sha}"
        )

    candidate_integrity = _probe_db_integrity(candidate_db)
    if candidate_integrity.lower() != "ok":
        raise RuntimeError(
            f"repair candidate integrity_check is not ok: {candidate_integrity}"
        )

    if not yes_flag and not force:
        # In CLI context the caller should have prompted interactively.
        # When called from Python API without yes_flag, raise.
        raise RuntimeError(
            "approve_swap_board requires explicit confirmation (yes_flag=True or force=True)"
        )

    maintenance = is_maintenance_mode(slug)
    running = _running_task_count(slug)
    if not force:
        if not maintenance:
            raise RuntimeError(
                "approve_swap_board requires maintenance mode unless force=True"
            )
        if running:
            raise RuntimeError(
                f"approve_swap_board refused: board has {running} running task(s); use force=True only after operator review"
            )

    # Pre-swap backup: use the same SQLite-aware backup primitive as the CLI,
    # falling back to raw forensic copy only when forced and SQLite cannot open.
    pre_swap_dir = db_path.parent / "pre-swaps"
    pre_swap_dir.mkdir(parents=True, exist_ok=True)
    pre_swap_backup = None
    if db_path.exists():
        pre_swap_backup = backup_board(
            slug,
            dest_dir=pre_swap_dir,
            label="pre-swap",
            force=True,
        )

    # Copy candidate into the destination directory first so os.replace is an
    # atomic same-filesystem rename, and preserve the candidate as evidence.
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=str(db_path.parent), prefix=f"{db_path.name}.swap.", suffix=".tmp"
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_path_str)
    try:
        shutil.copy2(str(candidate_db), str(tmp_path))
        # A candidate DB must not be paired with stale WAL/SHM sidecars from
        # the prior live DB. Remove them immediately before and after replace.
        for sidecar in _sidecar_paths(db_path):
            sidecar.unlink(missing_ok=True)
        os.replace(str(tmp_path), str(db_path))
        for sidecar in _sidecar_paths(db_path):
            sidecar.unlink(missing_ok=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    post_integrity = _probe_db_integrity(db_path)
    if post_integrity.lower() != "ok":
        raise RuntimeError(f"post-swap integrity_check is not ok: {post_integrity}")

    return {
        "board": slug,
        "swapped": True,
        "pre_swap_backup": str(pre_swap_backup) if pre_swap_backup else "",
        "candidate_db": str(candidate_db),
        "candidate_db_sha256": actual_sha,
        "post_swap_integrity_check": post_integrity,
        "maintenance_mode": maintenance,
        "running_tasks": running,
    }


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def set_maintenance_mode(
    board: str,
    enabled: bool,
    *,
    reason: str = "",
) -> dict:
    """Toggle maintenance mode in board metadata.

    Delegates to :func:`kanban_db.set_board_maintenance` so that the
    dispatcher, health surface, and CLI all read/write the same canonical
    ``maintenance`` key.
    """
    return kb.set_board_maintenance(board, enabled, reason=reason)


def is_maintenance_mode(board: str) -> bool:
    """Read maintenance mode from board metadata.

    Delegates to :func:`kanban_db.board_in_maintenance` so that the
    dispatcher, health surface, and CLI all agree on the same canonical
    ``maintenance`` key.
    """
    return kb.board_in_maintenance(board)


# ---------------------------------------------------------------------------
# Corruption injection helper (test-only)
# ---------------------------------------------------------------------------

def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _live_kanban_roots() -> tuple[Path, ...]:
    candidates = [
        Path.home().expanduser().resolve() / ".hermes",
        Path.home().expanduser().resolve() / ".hermes" / "kanban",
        Path("/home/openclaw/.hermes"),
        Path("/home/openclaw/.hermes/kanban"),
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        resolved = root.resolve()
        for path in (resolved, resolved / "boards"):
            key = str(path)
            if key not in seen:
                seen.add(key)
                roots.append(path)
    return tuple(roots)


def inject_corruption(db_path: Path, offset: int = 8192, length: int = 512) -> None:
    """Overwrite bytes in a non-live DB file to simulate page damage.

    This helper is intentionally fail-closed because it writes arbitrary bytes.
    It is for fixture/sandbox DBs only, never the operator's live Kanban root.
    """
    resolved = db_path.expanduser().resolve()
    for live_root in _live_kanban_roots():
        if _is_relative_to(resolved, live_root):
            raise RuntimeError(
                f"Refusing to inject corruption into live Kanban path: {resolved}"
            )

    data = resolved.read_bytes()
    corrupted = bytearray(data)
    for i in range(offset, min(offset + length, len(corrupted))):
        corrupted[i] = (corrupted[i] + 1) % 256
    resolved.write_bytes(bytes(corrupted))
