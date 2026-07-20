#!/usr/bin/env python3
"""Backfill legacy Hermes Kanban ``block_kind`` values conservatively.

The command examines one explicitly named board. It is dry-run by default;
``--apply`` is required to write classifications. Unknown legacy blocks remain
NULL so existing needs-input-safe behavior is preserved.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

VALID_BLOCK_KINDS = {
    "dependency",
    "needs_input",
    "capability",
    "transient",
    "infra",
    "budget",
}

ERROR_TYPE_TO_KIND = {
    "provider_quota": "infra",
    "provider_auth": "infra",
    "model_missing": "infra",
    "rate_limit": "transient",
    "budget_exhausted": "budget",
}

BUDGET_RE = re.compile(
    r"(?:iteration|turn)\s+budget\s+exhausted|budget exhausted\s*\(\d+/\d+\)",
    re.IGNORECASE,
)
RATE_LIMIT_RE = re.compile(r"\b429\b|rate.?limit", re.IGNORECASE)
INFRA_RE = re.compile(
    r"credential|api[ _-]?key|unauthorized|\b401\b|\b403\b|\b404\b|notfound"
    r"|unknown provider|quota|billing|usage limit|no such model"
    r"|model .* not found|spawn_failed|exec format error|provider unavailable",
    re.IGNORECASE,
)


def _payload(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _payload_text(payload: dict[str, Any]) -> str:
    return " ".join(
        str(payload.get(key) or "")
        for key in ("reason", "detail", "error", "summary", "fingerprint")
    ).strip()


def classify_block(
    *,
    blocked_payload: dict[str, Any],
    gave_up_payload: dict[str, Any],
    last_failure_error: str | None,
    block_fingerprint: str | None,
) -> tuple[str | None, str]:
    """Return a conservative ``(kind, evidence)`` classification."""
    explicit_kind = blocked_payload.get("kind")
    if explicit_kind in VALID_BLOCK_KINDS:
        return str(explicit_kind), _payload_text(blocked_payload)

    for payload in (blocked_payload, gave_up_payload):
        error_type = payload.get("error_type")
        mapped = ERROR_TYPE_TO_KIND.get(str(error_type))
        if mapped:
            return mapped, _payload_text(payload)

    blocked_text = _payload_text(blocked_payload)
    gave_up_text = _payload_text(gave_up_payload)
    evidence = " ".join(
        part
        for part in (
            blocked_text,
            gave_up_text,
            last_failure_error or "",
            block_fingerprint or "",
        )
        if part
    )
    if BUDGET_RE.search(evidence):
        return "budget", blocked_text or gave_up_text or evidence
    if RATE_LIMIT_RE.search(evidence):
        return "transient", blocked_text or gave_up_text or evidence
    if INFRA_RE.search(evidence):
        return "infra", blocked_text or gave_up_text or evidence
    return None, blocked_text or gave_up_text or last_failure_error or block_fingerprint or ""


def _latest_event_payloads(
    conn: sqlite3.Connection, task_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'gave_up') "
        "ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest.setdefault(row["kind"], _payload(row["payload"]))
    return latest.get("blocked", {}), latest.get("gave_up", {})


def backfill(db_path: Path, *, apply: bool = False) -> dict[str, Any]:
    if not db_path.is_file():
        raise FileNotFoundError(f"board database does not exist: {db_path}")

    if apply:
        conn = sqlite3.connect(db_path)
    else:
        # Enforce the dry-run contract at SQLite's connection boundary too, so
        # an accidental future write fails instead of silently changing a board.
        uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only = ON")
    conn.row_factory = sqlite3.Row
    try:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        required = {"id", "status", "block_kind", "last_failure_error"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                "board must be opened by the current Hermes migration first; "
                f"missing tasks columns: {', '.join(missing)}"
            )

        fingerprint_expr = (
            "block_fingerprint" if "block_fingerprint" in columns else "NULL"
        )
        candidates = conn.execute(
            "SELECT id, last_failure_error, "
            f"{fingerprint_expr} AS block_fingerprint "
            "FROM tasks WHERE status IN ('blocked', 'triage') "
            "AND block_kind IS NULL ORDER BY id"
        ).fetchall()

        classified: list[dict[str, str]] = []
        unclassified: list[dict[str, str]] = []
        for row in candidates:
            blocked_payload, gave_up_payload = _latest_event_payloads(conn, row["id"])
            kind, evidence = classify_block(
                blocked_payload=blocked_payload,
                gave_up_payload=gave_up_payload,
                last_failure_error=row["last_failure_error"],
                block_fingerprint=row["block_fingerprint"],
            )
            if kind is None:
                unclassified.append({"task_id": row["id"], "reason": evidence})
                continue
            classified.append(
                {"task_id": row["id"], "kind": kind, "evidence": evidence}
            )

        updated = 0
        if apply:
            with conn:
                for item in classified:
                    cursor = conn.execute(
                        "UPDATE tasks SET block_kind = ? "
                        "WHERE id = ? AND block_kind IS NULL",
                        (item["kind"], item["task_id"]),
                    )
                    updated += cursor.rowcount

        return {
            "database": str(db_path.resolve()),
            "mode": "apply" if apply else "dry-run",
            "counts": {
                "candidates": len(candidates),
                "classified": len(classified),
                "unclassified": len(unclassified),
                "updated": updated,
            },
            "classified": classified,
            "unclassified": unclassified,
        }
    finally:
        conn.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="one Kanban board DB")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="report classifications without writing (default)",
    )
    mode.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="write classified block_kind values",
    )
    parser.set_defaults(apply=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = backfill(args.db, apply=args.apply)
    except (FileNotFoundError, sqlite3.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
