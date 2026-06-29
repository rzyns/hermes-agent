"""SQLite-backed registry store for canonical cross-board edges.

Location: ``<KANBAN_HOME>/kanban/cross_board_dependencies.db``

Idempotent init with versioned migrations so repeated runs are safe.
Thread-safe via RLock around write paths; reads are direct SQLite which
is safe across threads for the same connection (but we keep short-lived
connections per operation to avoid cross-thread connection sharing).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

from plugins.kanban_cross_deps.models import (
    CrossBoardEdge,
    _parse_json_dict,
    _parse_json_list,
    _utc_now,
)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_FILENAME = "kanban/cross_board_dependencies.db"
CURRENT_SCHEMA_VERSION = 1

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS cross_board_edges (
    id TEXT PRIMARY KEY,
    parent_board TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    child_board TEXT NOT NULL,
    child_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    blocking INTEGER NOT NULL DEFAULT 1,
    required_parent_statuses TEXT NOT NULL DEFAULT '["done","archived"]',
    source TEXT NOT NULL,
    created_by TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT,
    UNIQUE(parent_board, parent_id, child_board, child_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_edges_child ON cross_board_edges(child_board, child_id);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON cross_board_edges(parent_board, parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON cross_board_edges(kind);

CREATE TABLE IF NOT EXISTS _registry_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class CrossBoardRegistry:
    """Canonical cross-board edge registry backed by SQLite."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path:
            self.path = Path(db_path)
        else:
            # Anchor to the shared kanban root (same as board DBs) so the
            # registry is visible across profiles that share boards.
            from hermes_cli.kanban_db import kanban_home
            self.path = kanban_home() / DEFAULT_REGISTRY_FILENAME
        self._init_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._inited: bool = False

    # -- internal helpers -----------------------------------------------------

    def _ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self):
        """Yield a short-lived SQLite connection with row factory."""
        self._ensure_dir()
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _read_conn(self):
        """Yield a read-only connection that does NOT create dirs or schema.

        If the DB file does not exist, yields None so callers can treat it
        as an empty registry.
        """
        if not self.path.exists():
            yield None
            return
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Idempotent schema creation and version bookkeeping."""
        if self._inited:
            return
        with self._init_lock:
            if self._inited:
                return
            self._ensure_dir()
            with self._conn() as conn:
                conn.executescript(_INIT_SQL)
                conn.execute(
                    """
                    INSERT INTO _registry_meta(key, value)
                    VALUES('schema_version', ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (str(CURRENT_SCHEMA_VERSION),),
                )
                conn.commit()
            self._inited = True

    def _ensure_init(self) -> None:
        if not self._inited:
            self._init_schema()

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> CrossBoardEdge:
        return CrossBoardEdge.from_row(dict(row))

    @staticmethod
    def _edge_to_row(edge: CrossBoardEdge) -> dict[str, Any]:
        return {
            "id": edge.id,
            "parent_board": edge.parent_board,
            "parent_id": edge.parent_id,
            "child_board": edge.child_board,
            "child_id": edge.child_id,
            "kind": edge.kind,
            "blocking": 1 if edge.blocking else 0,
            "required_parent_statuses": json.dumps(edge.required_parent_statuses),
            "source": edge.source,
            "created_by": edge.created_by,
            "created_at": int(edge.created_at.timestamp()) if edge.created_at else int(_utc_now().timestamp()),
            "updated_at": int(edge.updated_at.timestamp()) if edge.updated_at else int(_utc_now().timestamp()),
            "metadata": json.dumps(edge.metadata) if edge.metadata else None,
        }

    # -- cycle guard ----------------------------------------------------------

    def _would_create_cycle(
        self,
        parent_board: str,
        parent_id: str,
        child_board: str,
        child_id: str,
    ) -> bool:
        """Return True if adding a blocking edge would create a cycle.

        Walks both existing cross-board edges *and* local ``task_links`` so
        that a new cross-board edge cannot close a cycle through already
        existing board-local parent/child links.  This is the same graph
        semantics used by diagnostics and is stricter than the old pure
        cross-board guard.
        """
        from hermes_cli import kanban_db as _kb

        start: tuple[str, str] = (child_board, child_id)
        target: tuple[str, str] = (parent_board, parent_id)
        if start == target:
            return True

        seen: set[tuple[str, str]] = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)

            board, task_id = node
            # 1) Cross-board outgoing edges
            if self.path.exists():
                with self._read_conn() as conn:
                    if conn is not None:
                        cur = conn.execute(
                            "SELECT child_board, child_id FROM cross_board_edges "
                            "WHERE parent_board = ? AND parent_id = ? AND blocking = 1",
                            (board, task_id),
                        )
                        for r in cur.fetchall():
                            dst = (r["child_board"], r["child_id"])
                            if dst not in seen:
                                stack.append(dst)

            # 2) Local outgoing edges from the board-local task_links table
            try:
                conn = _kb.connect(board=board)
                try:
                    rows = conn.execute(
                        "SELECT child_id FROM task_links WHERE parent_id = ?",
                        (task_id,),
                    ).fetchall()
                    for r in rows:
                        dst = (board, r["child_id"])
                        if dst not in seen:
                            stack.append(dst)
                finally:
                    conn.close()
            except Exception:
                pass
        return False

    # -- public primitives ----------------------------------------------------

    def add(
        self,
        *,
        parent_board: str,
        parent_id: str,
        child_board: str,
        child_id: str,
        kind: str,
        blocking: bool = True,
        required_parent_statuses: list[str] | None = None,
        source: str = "canonical",
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
        edge_id: str | None = None,
        reject_cycle: bool = True,
    ) -> CrossBoardEdge:
        """Add a canonical edge.  Raises ValueError on invalid kind or cycle.
        Returns the persisted edge (with generated id and timestamps).
        """
        if blocking and reject_cycle:
            if self._would_create_cycle(parent_board, parent_id, child_board, child_id):
                raise ValueError(
                    f"Adding edge ({parent_board}/{parent_id}) -> ({child_board}/{child_id}) "
                    f"kind={kind} would create a blocking cycle"
                )

        self._ensure_init()
        now = _utc_now()
        edge = CrossBoardEdge(
            id=edge_id or _new_uuid(),
            parent_board=parent_board,
            parent_id=parent_id,
            child_board=child_board,
            child_id=child_id,
            kind=kind,
            blocking=blocking,
            required_parent_statuses=required_parent_statuses or ["done", "archived"],
            source=source,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        row = self._edge_to_row(edge)
        with self._write_lock:
            with self._conn() as conn:
                try:
                    conn.execute(
                        """
                        INSERT INTO cross_board_edges(
                            id, parent_board, parent_id, child_board, child_id,
                            kind, blocking, required_parent_statuses, source,
                            created_by, created_at, updated_at, metadata
                        ) VALUES(
                            :id, :parent_board, :parent_id, :child_board, :child_id,
                            :kind, :blocking, :required_parent_statuses, :source,
                            :created_by, :created_at, :updated_at, :metadata
                        )
                        """,
                        row,
                    )
                    conn.commit()
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"Edge already exists for ({parent_board}/{parent_id}) -> ({child_board}/{child_id}) kind={kind}"
                    ) from exc
        return edge

    def remove(self, edge_id: str) -> bool:
        """Remove a canonical edge by id.  Returns True if row existed."""
        self._ensure_init()
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute("DELETE FROM cross_board_edges WHERE id = ?", (edge_id,))
                conn.commit()
                return cur.rowcount > 0

    def remove_by_composite(
        self,
        *,
        parent_board: str,
        parent_id: str,
        child_board: str,
        child_id: str,
        kind: str,
    ) -> bool:
        """Remove by unique composite key."""
        self._ensure_init()
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    """
                    DELETE FROM cross_board_edges
                    WHERE parent_board = ? AND parent_id = ?
                      AND child_board = ? AND child_id = ?
                      AND kind = ?
                    """,
                    (parent_board, parent_id, child_board, child_id, kind),
                )
                conn.commit()
                return cur.rowcount > 0

    def get(self, edge_id: str) -> CrossBoardEdge | None:
        """Fetch a single edge by id.  Does not create the registry DB."""
        with self._read_conn() as conn:
            if conn is None:
                return None
            cur = conn.execute("SELECT * FROM cross_board_edges WHERE id = ?", (edge_id,))
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def list_edges(
        self,
        *,
        child_board: str | None = None,
        child_id: str | None = None,
        parent_board: str | None = None,
        parent_id: str | None = None,
        kind: str | None = None,
        blocking: bool | None = None,
        source: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[CrossBoardEdge]:
        """List/filter edges.  All filters are ANDed.  Does not create the DB."""
        clauses: list[str] = []
        params: list[Any] = []
        if child_board is not None:
            clauses.append("child_board = ?")
            params.append(child_board)
        if child_id is not None:
            clauses.append("child_id = ?")
            params.append(child_id)
        if parent_board is not None:
            clauses.append("parent_board = ?")
            params.append(parent_board)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if blocking is not None:
            clauses.append("blocking = ?")
            params.append(1 if blocking else 0)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"""
            SELECT * FROM cross_board_edges
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        with self._read_conn() as conn:
            if conn is None:
                return []
            cur = conn.execute(sql, params)
            return [self._row_to_edge(row) for row in cur.fetchall()]

    def count(
        self,
        *,
        child_board: str | None = None,
        child_id: str | None = None,
        parent_board: str | None = None,
        parent_id: str | None = None,
        kind: str | None = None,
        blocking: bool | None = None,
        source: str | None = None,
    ) -> int:
        """Return matching edge count.  Does not create the DB."""
        clauses: list[str] = []
        params: list[Any] = []
        if child_board is not None:
            clauses.append("child_board = ?")
            params.append(child_board)
        if child_id is not None:
            clauses.append("child_id = ?")
            params.append(child_id)
        if parent_board is not None:
            clauses.append("parent_board = ?")
            params.append(parent_board)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if blocking is not None:
            clauses.append("blocking = ?")
            params.append(1 if blocking else 0)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT COUNT(*) FROM cross_board_edges WHERE {where}"
        with self._read_conn() as conn:
            if conn is None:
                return 0
            cur = conn.execute(sql, params)
            return int(cur.fetchone()[0])

    def update_metadata(
        self,
        edge_id: str,
        metadata: dict[str, Any],
        merge: bool = True,
    ) -> CrossBoardEdge | None:
        """Update edge metadata.  If merge=True, deep-merge over existing.
        Returns the updated edge, or None if not found.
        """
        self._ensure_init()
        with self._write_lock:
            with self._conn() as conn:
                cur = conn.execute(
                    "SELECT metadata FROM cross_board_edges WHERE id = ?",
                    (edge_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                existing = _parse_json_dict(row["metadata"])
                if merge:
                    merged = {**existing, **metadata}
                else:
                    merged = dict(metadata)
                now = int(_utc_now().timestamp())
                conn.execute(
                    "UPDATE cross_board_edges SET metadata = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(merged), now, edge_id),
                )
                conn.commit()
        return self.get(edge_id)

    def update_blocking(
        self,
        edge_id: str,
        blocking: bool,
    ) -> CrossBoardEdge | None:
        """Update the blocking flag for an edge.  Returns updated edge or None.

        Turning a non-blocking edge into a blocking edge is a semantic class
        identical to adding a new blocking edge: it may create a cycle.  Guard
        against that by running the same cycle check before committing.
        """
        self._ensure_init()
        # Guard: promoting to blocking must not introduce a cycle.
        if blocking:
            edge = self.get(edge_id)
            if edge is not None and not edge.blocking:
                if self._would_create_cycle(
                    parent_board=edge.parent_board,
                    parent_id=edge.parent_id,
                    child_board=edge.child_board,
                    child_id=edge.child_id,
                ):
                    raise ValueError(
                        f"Promoting edge {edge_id} to blocking would create a cycle"
                    )
        with self._write_lock:
            with self._conn() as conn:
                now = int(_utc_now().timestamp())
                cur = conn.execute(
                    "UPDATE cross_board_edges SET blocking = ?, updated_at = ? WHERE id = ?",
                    (1 if blocking else 0, now, edge_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
        return self.get(edge_id)

    def schema_version(self) -> int:
        """Return the current schema version from _registry_meta.
        Does not create the DB.
        """
        with self._read_conn() as conn:
            if conn is None:
                return 0
            cur = conn.execute(
                "SELECT value FROM _registry_meta WHERE key = 'schema_version'"
            )
            row = cur.fetchone()
            if row is None:
                return 0
            try:
                return int(row["value"])
            except Exception:
                return 0

    def reset(self) -> None:
        """Drop all edges and meta (for tests).  Destructive."""
        self._ensure_init()
        with self._write_lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM cross_board_edges")
                conn.execute("DELETE FROM _registry_meta")
                conn.commit()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return "x" + uuid.uuid4().hex[:15]
