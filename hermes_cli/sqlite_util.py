"""Shared SQLite primitives for the small per-profile / board stores.

The projects and kanban stores open WAL SQLite files with the same two
primitives — an idempotent column-add migration and an IMMEDIATE write
transaction. One definition here keeps the two stores from drifting.
"""

from __future__ import annotations

import contextlib
import sqlite3


def add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    """``ALTER TABLE <table> ADD COLUMN <ddl>``, idempotent across races.

    Returns ``True`` when this call added the column. Swallows the
    ``duplicate column name`` error a concurrent migrator may have run first
    (issue #21708). ``column`` is the human-readable name for the call site;
    ``ddl`` carries the actual definition.
    """
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
        return True
    except sqlite3.OperationalError as exc:
        if "duplicate column name" in str(exc).lower():
            return False
        raise


def create_unique_index_if_missing(
    conn: sqlite3.Connection,
    index_name: str,
    table: str,
    columns: str,
) -> bool:
    """Create a unique index idempotently, swallowing the already-exists case.

    Caller is responsible for deduplicating rows that would violate the unique
    constraint before calling this helper; this function does NOT repair data.
    Returns ``True`` when the index was created by this call.
    """
    sql = f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table} ({columns})"
    try:
        conn.execute(sql)
        return True
    except sqlite3.OperationalError as exc:
        lowered = str(exc).lower()
        if "already exists" in lowered:
            return False
        raise


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """An IMMEDIATE write transaction: at most one concurrent writer wins.

    The explicit ROLLBACK is guarded so a SQLite auto-rollback (no active
    transaction left under EIO / lock contention / corruption) cannot shadow
    the original exception with a spurious rollback error.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise
    else:
        conn.execute("COMMIT")
