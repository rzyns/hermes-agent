"""Read-only register-health checks for the Attention Intake link-drop path.

This module is intentionally separate from ``kanban_intake_link.py`` so:
- the write path stays focused on creation / provisional update;
- the health-check surface can be imported by dashboard, CLI, tests,
  and watchdogs without pulling creation logic.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

URL_RE = __import__("re").compile(r"https?://[^\s)\]}>\"`']+")

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact_root(hermes_home: Path) -> Path:
    return hermes_home / "artifacts" / "attention-intake"


def _register_paths(hermes_home: Path) -> tuple[Path, Path]:
    root = _artifact_root(hermes_home)
    return root / "register.md", root / "register.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("JSON decode error on %s line %d", path, line_no)
    return rows


def _extract_url(text: str | None) -> str | None:
    if not text:
        return None
    match = URL_RE.search(text)
    return match.group(0).rstrip(".") if match else None


# ---------------------------------------------------------------------------
# Core checkers
# ---------------------------------------------------------------------------


def check_register_for_task(
    task_id: str,
    task_body: str | None,
    *,
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    """Return structured register-health for a single intake-link task.

    Looks for:
    - provisional ``intake_link_created`` JSONL entry with matching task_id
    - full register entry (JSONL or MD) covering the same URL.
    - task body pointer back to register files.

    Returns a dict ready for JSON serialization or dashboard embedding.
    """
    home = hermes_home or Path(__import__("os").environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    home = Path(home)

    md_path, jsonl_path = _register_paths(home)
    rows = _load_jsonl(jsonl_path)

    url = _extract_url(task_body or "")
    result: dict[str, Any] = {
        "task_id": task_id,
        "url": url,
        "register_md_exists": md_path.exists(),
        "register_md_path": str(md_path),
        "register_jsonl_exists": jsonl_path.exists(),
        "register_jsonl_path": str(jsonl_path),
        "register_row_count": len(rows),
    }

    # Look for provisional entry
    provisional = [r for r in rows if r.get("task_id") == task_id and r.get("event") == "intake_link_created"]
    result["provisional_entries"] = provisional
    result["has_provisional_entry"] = bool(provisional)

    # Look for full register entry by URL or task_id
    full_by_url = [r for r in rows if r.get("url") == url and r.get("event") != "intake_link_created"] if url else []
    full_by_task = [r for r in rows if (r.get("source_task") == task_id or r.get("final_task") == task_id) and r.get("event") != "intake_link_created"]
    result["full_entries_by_url"] = len(full_by_url)
    result["full_entries_by_task"] = len(full_by_task)
    result["has_full_entry"] = bool(full_by_url or full_by_task)

    # Body contract check
    body_contract_ok = False
    if task_body:
        body_contract_ok = (
            "register.jsonl" in task_body
            and "register.md" in task_body
            and "needs_assessment" in task_body
        )
    result["body_contract_ok"] = body_contract_ok

    # Overall verdict
    verdict = "complete"
    if not result["has_provisional_entry"]:
        verdict = "missing_provisional"
    elif not result["has_full_entry"]:
        verdict = "provisional_only"
    if not body_contract_ok:
        verdict = "incomplete_body"
    result["verdict"] = verdict
    return result


def scan_board_for_health(
    conn: sqlite3.Connection,
    *,
    board: str = "attention-intake",
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    """Scan every intake-link task on ``board`` and report aggregate health.

    Returns ``{
        board, scanned_task_count,
        provisionally_registered_count, fully_registered_count,
        missing_provisional_count, provisional_only_count,
        incomplete_body_count, unknown_source_count,
        tasks: [check_register_for_task(...), ...]
    }``.
    """
    home = hermes_home or Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    home = Path(home)

    rows = conn.execute(
        "SELECT id, body, title, status FROM tasks WHERE title LIKE 'Link drop:%' ORDER BY created_at, id",
    ).fetchall()

    tasks: list[dict[str, Any]] = []
    for r in rows:
        health = check_register_for_task(r["id"], r["body"], hermes_home=home)
        health["status"] = r["status"]
        tasks.append(health)

    counts: dict[str, int] = {
        "provisionally_registered": sum(1 for t in tasks if t["has_provisional_entry"]),
        "fully_registered": sum(1 for t in tasks if t["has_full_entry"]),
        "missing_provisional": sum(1 for t in tasks if t["verdict"] == "missing_provisional"),
        "provisional_only": sum(1 for t in tasks if t["verdict"] == "provisional_only"),
        "incomplete_body": sum(1 for t in tasks if t["body_contract_ok"] is False),
    }

    return {
        "board": board,
        "scanned_task_count": len(tasks),
        "counts": counts,
        "tasks": tasks,
    }


def provisional_entry_count(
    *,
    hermes_home: Path | None = None,
) -> dict[str, Any]:
    """Quick count of provisional ``intake_link_created`` entries in the
    Attention Intake JSONL register, plus total register rows.
    """
    home = hermes_home or Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
    home = Path(home)
    _, jsonl_path = _register_paths(home)
    rows = _load_jsonl(jsonl_path)
    provisional = [r for r in rows if r.get("event") == "intake_link_created"]
    return {
        "provisional_count": len(provisional),
        "total_rows": len(rows),
        "register_jsonl_exists": jsonl_path.exists(),
        "register_jsonl_path": str(jsonl_path),
    }
