"""Append-only JSONL semantic event sidecar for Kanban disaster recovery.

Each board gets a ``kanban.events/`` directory alongside ``kanban.db``.
Events are written as JSONL lines with per-line SHA-256 integrity hashes
in a parallel ``.sha256`` file.  Rotation is by size (default 100 MB)
and/or daily.  All behaviour is gated by ``kanban.sidecar.enabled``.

This module is intentionally narrow: it does not replace SQLite or
the ``task_events`` table.  It is a durability amplifier that survives
DB corruption.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import hermes_cli.config as cfg_mod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIDECAR_DIR_NAME = "kanban.events"
CURRENT_FILE = "current.jsonl"
CURRENT_HASH_FILE = "current.jsonl.sha256"
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _sidecar_enabled() -> bool:
    return _kanban_cfg().get("sidecar", {}).get("enabled", False)


def _sidecar_max_bytes() -> int:
    return _kanban_cfg().get("sidecar", {}).get("max_bytes", 100 * 1024 * 1024)


def _sidecar_rotate_daily() -> bool:
    return _kanban_cfg().get("sidecar", {}).get("rotate_daily", True)


def _sidecar_sync_mode() -> str:
    return _kanban_cfg().get("sidecar", {}).get("sync_mode", "O_DSYNC")


def _sidecar_retention_days() -> int:
    return _kanban_cfg().get("sidecar", {}).get("retention_days", 90)


def _kanban_cfg() -> Dict[str, Any]:
    try:
        # Use readonly variant for hot-path; sidecar writes happen inside
        # a txn so the extra microsecond does not matter, but keeping it
        # consistent with other kanban_db reads is tidy.
        return cfg_mod.load_config_readonly().get("kanban") or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _sidecar_dir(board_dir: Path) -> Path:
    return board_dir / SIDECAR_DIR_NAME


def _current_jsonl(board_dir: Path) -> Path:
    return _sidecar_dir(board_dir) / CURRENT_FILE


def _current_hashes(board_dir: Path) -> Path:
    return _sidecar_dir(board_dir) / CURRENT_HASH_FILE


def _sidecar_for_db(db_path: Path) -> Tuple[Path, Path]:
    """Return (jsonl, hashes) paths for the sidecar of ``db_path``."""
    sdir = _sidecar_dir(db_path.parent)
    return sdir / CURRENT_FILE, sdir / CURRENT_HASH_FILE


# ---------------------------------------------------------------------------
# Hash / canonicalisation
# ---------------------------------------------------------------------------


def _canonical_json(obj: Dict[str, Any]) -> str:
    """Stable JSON for hashing: sorted keys, no extra whitespace,
    ``ensure_ascii=False``, UTF-8 encoded."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash_event(event: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of ``event`` *without* its ``hash`` key."""
    body = {k: v for k, v in event.items() if k != "hash"}
    canonical = _canonical_json(body)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verify_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a JSONL line and verify its embedded hash.

    Returns the parsed dict on success, ``None`` if the hash does not
    match (corruption / tampering).
    """
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    expected = event.get("hash")
    if not expected:
        return None
    computed = _hash_event(event)
    return event if computed == expected else None


# ---------------------------------------------------------------------------
# Sequence-number tracking
# ---------------------------------------------------------------------------


def _read_max_seq(board_dir: Path) -> int:
    """Scan ``current.jsonl`` for the highest ``seq`` seen so far."""
    path = _current_jsonl(board_dir)
    if not path.exists():
        return 0
    max_seq = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                max_seq = max(max_seq, event.get("seq", 0))
            except json.JSONDecodeError:
                continue
    return max_seq


# ---------------------------------------------------------------------------
# Low-level append (file descriptor, POSIX advisory lock)
# ---------------------------------------------------------------------------


def _open_append_fd(path: Path, sync_mode: str) -> int:
    """Open ``path`` for O_APPEND writing with appropriate durability."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if sync_mode == "O_DSYNC":
        flags |= os.O_DSYNC
    return os.open(str(path), flags, 0o644)


@contextmanager
def _locked_append(board_dir: Path):
    """Yield a writable sidecar state inside a POSIX advisory lock.

    The lock is on a sentinel ``.lock`` file inside the sidecar dir so
    that rotation (rename) does not disturb the lock.
    """
    sdir = _sidecar_dir(board_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    lock_path = sdir / ".lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def _needs_rotation(board_dir: Path, max_bytes: int, rotate_daily: bool) -> bool:
    jsonl = _current_jsonl(board_dir)
    if not jsonl.exists():
        return False
    if jsonl.stat().st_size > max_bytes:
        return True
    if rotate_daily:
        # Rotate if the file's mtime is from a different UTC day than now.
        mtime = jsonl.stat().st_mtime
        if time.gmtime(mtime).tm_yday != time.gmtime().tm_yday:
            return True
    return False


def _rotate(board_dir: Path) -> None:
    """Rename ``current.jsonl`` to a date-stamped segment and start fresh.

    Called inside the advisory lock.  Does nothing if the current file is
    empty or missing.
    """
    jsonl = _current_jsonl(board_dir)
    hashes = _current_hashes(board_dir)
    if not jsonl.exists() or jsonl.stat().st_size == 0:
        # Start fresh hash file too if it exists and jsonl is missing
        if hashes.exists():
            hashes.unlink()
        return

    ts = time.strftime("%Y-%m-%d")
    sdir = _sidecar_dir(board_dir)
    # Disambiguate collisions: 2026-06-11_1.jsonl, _2, etc.
    n = 0
    while True:
        suffix = f"_{n}" if n else ""
        target_jsonl = sdir / f"{ts}{suffix}.jsonl"
        target_hashes = sdir / f"{ts}{suffix}.jsonl.sha256"
        if not target_jsonl.exists():
            break
        n += 1

    # Final hash sweep: recompute hashes for the closing segment.
    _rewrite_hash_file(jsonl, hashes)

    os.rename(str(jsonl), str(target_jsonl))
    if hashes.exists():
        os.rename(str(hashes), str(target_hashes))


# ---------------------------------------------------------------------------
# Hash-file maintenance
# ---------------------------------------------------------------------------


def _rewrite_hash_file(jsonl: Path, hashes: Path) -> None:
    """Recompute ``.sha256`` from ``jsonl`` from scratch."""
    if not jsonl.exists():
        if hashes.exists():
            hashes.unlink()
        return
    with open(jsonl, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    digests = []
    for line in lines:
        if not line.strip():
            continue
        event = _verify_line(line)
        if event is None:
            # Keep a placeholder so line count stays aligned; verifier
            # will flag this later.
            digests.append("INVALID")
        else:
            digests.append(event["hash"])
    with open(hashes, "w", encoding="utf-8") as fh:
        for d in digests:
            fh.write(d + "\n")


# ---------------------------------------------------------------------------
# Public API: append_event
# ---------------------------------------------------------------------------


def append_event(
    board_dir: Path,
    task_id: str,
    kind: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Append a semantic event to the sidecar for ``board_dir``.

    This is a no-op unless ``kanban.sidecar.enabled`` is ``True``.
    The caller (``kanban_db.py``) is responsible for deciding *whether*
    to call this inside its transaction.
    """
    if not _sidecar_enabled():
        return

    seq = _read_max_seq(board_dir) + 1
    event: Dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "ts": int(time.time()),
        "seq": seq,
        "kind": kind,
        "task_id": task_id,
        "payload": payload,
    }
    event["hash"] = _hash_event(event)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    with _locked_append(board_dir):
        # Rotation check *inside* the lock so two writers cannot both
        # rotate concurrently.
        if _needs_rotation(board_dir, _sidecar_max_bytes(), _sidecar_rotate_daily()):
            _rotate(board_dir)

        jsonl_path, hashes_path = _sidecar_for_db(board_dir / "kanban.db")
        sdir = _sidecar_dir(board_dir)
        sdir.mkdir(parents=True, exist_ok=True)

        fd = _open_append_fd(jsonl_path, _sidecar_sync_mode())
        try:
            os.write(fd, (line + "\n").encode("utf-8"))
            if _sidecar_sync_mode() != "O_DSYNC":
                os.fsync(fd)
        finally:
            os.close(fd)

        # Append hash in a second fd so hash file stays line-aligned.
        fd_h = _open_append_fd(hashes_path, _sidecar_sync_mode())
        try:
            os.write(fd_h, (event["hash"] + "\n").encode("utf-8"))
            if _sidecar_sync_mode() != "O_DSYNC":
                os.fsync(fd_h)
        finally:
            os.close(fd_h)


# ---------------------------------------------------------------------------
# Rebuild helpers
# ---------------------------------------------------------------------------


def iter_segments(board_dir: Path) -> Iterator[Path]:
    """Yield sidecar segment paths in chronological order (oldest first).

    Includes ``current.jsonl`` if it exists.
    """
    sdir = _sidecar_dir(board_dir)
    if not sdir.exists():
        return
    # Rotated segments: YYYY-MM-DD.jsonl or YYYY-MM-DD_N.jsonl
    rotated = sorted(
        p for p in sdir.iterdir()
        if p.suffix == ".jsonl" and p.name != CURRENT_FILE
    )
    for p in rotated:
        yield p
    current = sdir / CURRENT_FILE
    if current.exists():
        yield current


def verify_segment(jsonl: Path, hashes: Optional[Path] = None) -> Tuple[int, int, List[str]]:
    """Verify a segment file.  Returns (lines_ok, lines_bad, warnings).

    If ``hashes`` is not provided, looks for the sibling ``.sha256``.
    """
    if hashes is None:
        hashes = jsonl.with_suffix(jsonl.suffix + ".sha256")
    ok = 0
    bad = 0
    warnings: List[str] = []
    hash_lines: List[str] = []
    if hashes.exists():
        with open(hashes, "r", encoding="utf-8") as fh:
            hash_lines = [ln.strip() for ln in fh.readlines()]
    with open(jsonl, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            event = _verify_line(line)
            if event is None:
                bad += 1
                warnings.append(f"line {i}: hash mismatch or invalid JSON")
                continue
            if hash_lines and i - 1 < len(hash_lines) and hash_lines[i - 1] != event["hash"]:
                bad += 1
                warnings.append(f"line {i}: hash-file mismatch")
                continue
            ok += 1
    return ok, bad, warnings


@dataclass
class RebuildReport:
    events_replayed: int = 0
    tasks_created: int = 0
    tasks_updated: int = 0
    comments_added: int = 0
    runs_ended: int = 0
    warnings: List[str] = field(default_factory=list)
    max_seq: int = 0


def rebuild_board(
    board_dir: Path,
    target_db_path: Path,
    *,
    segments: Optional[List[Path]] = None,
) -> RebuildReport:
    """Replay sidecar events into a fresh ``kanban.db``.

    **Idempotency:** ``task_created`` events are skipped if the task
    already exists.  Events with ``seq <= last_processed_seq`` are
    skipped on resume.

    **Text loss:** comment bodies and run summaries are NOT stored in
    the sidecar; they become empty placeholders.  This is by design.

    Returns a :class:`RebuildReport`.
    """
    from hermes_cli.kanban_db import init_db, connect

    report = RebuildReport()
    init_db(target_db_path)
    last_seq = 0

    if segments is None:
        segments = list(iter_segments(board_dir))

    for seg in segments:
        with open(seg, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                event = _verify_line(line)
                if event is None:
                    report.warnings.append(f"skip unverifiable line in {seg.name}")
                    continue
                seq = event.get("seq", 0)
                if seq <= last_seq:
                    report.warnings.append(f"skip duplicate seq {seq}")
                    continue
                last_seq = seq
                report.max_seq = max(report.max_seq, seq)

                _apply_event_to_db(target_db_path, event, report)
                report.events_replayed += 1

    return report


def _apply_event_to_db(db_path: Path, event: Dict[str, Any], report: RebuildReport) -> None:
    """Apply a single sidecar event to the board DB."""
    import sqlite3

    kind = event["kind"]
    task_id = event.get("task_id", "")
    payload = event.get("payload") or {}
    ts = event.get("ts", int(time.time()))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if kind == "task_created":
            existing = conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if not existing:
                title = payload.get("title", "")
                status = payload.get("status", "todo")
                assignee = payload.get("assignee")
                tenant = payload.get("tenant")
                conn.execute(
                    "INSERT INTO tasks (id, title, status, assignee, tenant, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (task_id, title, status, assignee, tenant, ts),
                )
                report.tasks_created += 1
        elif kind == "task_status_changed":
            new_status = payload.get("new", "")
            if new_status:
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    (new_status, task_id),
                )
                report.tasks_updated += 1
        elif kind == "task_assigned":
            assignee = payload.get("assignee")
            if assignee:
                conn.execute(
                    "UPDATE tasks SET assignee = ? WHERE id = ?",
                    (assignee, task_id),
                )
                report.tasks_updated += 1
        elif kind == "commented":
            author = payload.get("author", "unknown")
            body_len = payload.get("len", 0)
            # Insert a placeholder — sidecar does not store full text.
            body = f"[recovered placeholder; original length {body_len}]"
            comment_id = f"c_{ts}_{task_id}"
            conn.execute(
                "INSERT INTO task_comments (id, task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (comment_id, task_id, author, body, ts),
            )
            report.comments_added += 1
        elif kind == "run_ended":
            outcome = payload.get("outcome", "")
            summary = payload.get("summary")
            error = payload.get("error")
            run_id = payload.get("run_id")
            if run_id:
                conn.execute(
                    "INSERT OR IGNORE INTO task_runs "
                    "(id, task_id, status, outcome, summary, error, started_at, ended_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, task_id, outcome, outcome, summary, error, ts, ts),
                )
                report.runs_ended += 1
        # Board-level events (heartbeat, rotated, etc.) are intentionally
        # no-ops during rebuild; they are metadata about the sidecar itself.
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Maintenance: GC old rotated segments
# ---------------------------------------------------------------------------


def gc_old_segments(board_dir: Path, retention_days: Optional[int] = None) -> int:
    """Delete rotated segments older than ``retention_days``.  Returns number
    of files removed."""
    if retention_days is None:
        retention_days = _sidecar_retention_days()
    cutoff = time.time() - (retention_days * 86400)
    sdir = _sidecar_dir(board_dir)
    if not sdir.exists():
        return 0
    removed = 0
    for p in sdir.iterdir():
        if p.name in (CURRENT_FILE, CURRENT_HASH_FILE, ".lock"):
            continue
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed
