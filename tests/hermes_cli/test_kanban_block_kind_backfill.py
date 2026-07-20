"""Schema and dry-run coverage for the block-kind taxonomy backfill."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


_NEW_BLOCK_COLUMNS = {
    "block_fingerprint": "TEXT",
    "block_deadline": "INTEGER",
    "block_retry_after": "INTEGER",
    "block_error_type": "TEXT",
}


def _create_legacy_board(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        );
        CREATE TABLE task_links (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id)
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        );
        INSERT INTO tasks (id, title, status, created_at)
        VALUES ('t_legacy', 'legacy block', 'blocked', 1);
        """
    )
    conn.commit()
    conn.close()


def test_block_metadata_columns_migrate_idempotently_and_stay_nullable(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_board(db_path)

    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path):
        pass
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        columns = {
            row["name"]: row
            for row in conn.execute("PRAGMA table_info(tasks)")
        }
        legacy = conn.execute(
            "SELECT block_fingerprint, block_deadline, block_retry_after, "
            "block_error_type FROM tasks WHERE id = 't_legacy'"
        ).fetchone()

    for name, declared_type in _NEW_BLOCK_COLUMNS.items():
        assert columns[name]["type"] == declared_type
        assert columns[name]["notnull"] == 0
    assert tuple(legacy) == (None, None, None, None)


@pytest.mark.parametrize("kind", ["infra", "budget"])
def test_block_task_accepts_new_block_kinds(tmp_path, monkeypatch, kind):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=f"{kind} blocker", assignee="worker")
        assert kb.block_task(conn, task_id, reason="classified", kind=kind)
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "blocked"
    assert task.block_kind == kind


def test_backfill_defaults_to_dry_run_and_reports_unclassified(tmp_path):
    db_path = tmp_path / "fixture-board.db"
    with kb.connect(db_path) as conn:
        rows = [
            ("t_payload", "payload kind", "blocked", None, None),
            ("t_budget", "budget", "blocked", None, None),
            ("t_fingerprint", "credential", "blocked", None, "API key missing"),
            ("t_unknown", "unknown", "blocked", None, None),
            ("t_existing", "existing", "blocked", "transient", None),
        ]
        conn.executemany(
            "INSERT INTO tasks "
            "(id, title, status, created_at, block_kind, last_failure_error) "
            "VALUES (?, ?, ?, 1, ?, ?)",
            rows,
        )
        events = [
            ("t_payload", "blocked", {"reason": "provider unavailable", "kind": "infra"}),
            ("t_budget", "gave_up", {"reason": "turn budget exhausted (3/3)"}),
            ("t_unknown", "blocked", {"reason": "requires further investigation"}),
        ]
        conn.executemany(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, 2)",
            [(task_id, kind, json.dumps(payload)) for task_id, kind, payload in events],
        )
        conn.commit()

    script = Path(__file__).resolve().parents[2] / "scripts" / "backfill_kanban_block_kinds.py"
    result = subprocess.run(
        [sys.executable, str(script), "--db", str(db_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)

    assert report["mode"] == "dry-run"
    assert report["counts"] == {
        "candidates": 4,
        "classified": 3,
        "unclassified": 1,
        "updated": 0,
    }
    assert {item["task_id"]: item["kind"] for item in report["classified"]} == {
        "t_payload": "infra",
        "t_budget": "budget",
        "t_fingerprint": "infra",
    }
    assert report["unclassified"] == [
        {"task_id": "t_unknown", "reason": "requires further investigation"}
    ]

    conn = sqlite3.connect(db_path)
    stored = dict(conn.execute("SELECT id, block_kind FROM tasks"))
    conn.close()
    assert stored == {
        "t_payload": None,
        "t_budget": None,
        "t_fingerprint": None,
        "t_unknown": None,
        "t_existing": "transient",
    }

    first_apply = subprocess.run(
        [sys.executable, str(script), "--db", str(db_path), "--apply"],
        check=True,
        capture_output=True,
        text=True,
    )
    second_apply = subprocess.run(
        [sys.executable, str(script), "--db", str(db_path), "--apply"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(first_apply.stdout)["counts"]["updated"] == 3
    assert json.loads(second_apply.stdout)["counts"] == {
        "candidates": 1,
        "classified": 0,
        "unclassified": 1,
        "updated": 0,
    }
