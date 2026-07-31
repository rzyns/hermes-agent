"""t_fa5cf8ac: regression evidence for fail-closed temp-DB isolation.

Preserves the full t_0645f051 13-test mutation-isolation matrix and adds the
t_dae1e07e repair tests that close four additional gaps:

1. ``write_txn`` validates the actual SQLite main DB path from the connection
   (``PRAGMA database_list``), not the current env resolution.
2. ``clear_board_maintenance`` is guarded before writing board metadata.
3. ``delete_attachment`` refuses to unlink a ``stored_path`` outside the
   isolation root and keeps DB/file state consistent.
4. ``_resolve_worktree_workspace`` / ``resolve_workspace`` refuse to
   materialize a worktree outside the declared temp root.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Temp HERMES_HOME used as the isolation root."""
    home = tmp_path / "isolated_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ATTACHMENTS_ROOT", raising=False)
    # Clear per-process caches so env changes in this test file are observed.
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
    kb.init_db()
    return home


@pytest.fixture
def live_root(tmp_path):
    """Directory outside the isolation root that simulates a live board."""
    live = tmp_path / "live_board"
    live.mkdir()
    return live


# ---------------------------------------------------------------------------
# t_0645f051 core matrix: DB path guards
# ---------------------------------------------------------------------------


def test_connect_blocked_when_hermes_kanban_db_points_outside_root(
    isolated_home, live_root, monkeypatch
):
    """HERMES_KANBAN_DB outside the declared temp root is refused."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_root / "kanban.db"))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.connect()


def test_init_db_blocked_when_hermes_kanban_db_points_outside_root(
    isolated_home, live_root, monkeypatch
):
    """init_db also fails closed when the env pin targets an outside DB."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_root / "kanban.db"))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.init_db()


def test_write_txn_blocks_preopened_connection_after_env_switch(
    isolated_home, live_root, monkeypatch
):
    """A conn opened on an outside DB is blocked after the env pin flips in-root.

    Regression for 9746fbb285: that revision checked ``kanban_db_path()``
    (env-based resolution) inside ``write_txn``. Once ``HERMES_KANBAN_DB`` is
    switched to an in-root path, the env-based guard permitted the write even
    though the supplied connection object still pointed at the live outside DB.

    The fix (t_dae1e07e) makes ``write_txn`` check ``PRAGMA database_list`` on
    the connection itself, so this test MUST open the outside connection first,
    then explicitly set ``HERMES_KANBAN_DB`` to an in-root path, then enter the
    isolation envelope, and finally prove the outside DB is untouched.
    """
    outside_db = live_root / "kanban.db"
    inside_db = isolated_home / "kanban.db"

    # Open a connection whose SQLite main database is the outside file.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(outside_db))
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
    conn = kb.connect()
    try:
        # Explicitly switch the env pin to the in-root DB. The connection
        # object still points at outside_db, but env-based resolution now
        # reports the isolation-root path.
        monkeypatch.setenv("HERMES_KANBAN_DB", str(inside_db))
        kb._INITIALIZED_PATHS.clear()
        kb._INITIALIZED_PATH_FINGERPRINTS.clear()

        with kb.kanban_mutation_isolation(isolated_home):
            with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
                kb.create_task(conn, title="outside root", assignee="reviewer")
    finally:
        conn.close()

    # Prove the outside DB was not mutated by the refused operation.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(outside_db))
    kb._INITIALIZED_PATHS.clear()
    kb._INITIALIZED_PATH_FINGERPRINTS.clear()
    with kb.connect() as check:
        assert check.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0

    # Sanity: outside the envelope the same path writes fine.
    with kb.connect() as conn:
        kb.create_task(conn, title="outside no envelope", assignee="reviewer")


def test_mutations_allowed_under_isolation_root(isolated_home):
    """Inside the envelope, ordinary mutations to the temp root succeed."""
    with kb.kanban_mutation_isolation(isolated_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="fixture", assignee="reviewer")
            task = kb.get_task(conn, tid)

    assert task is not None
    assert task.title == "fixture"


# ---------------------------------------------------------------------------
# t_0645f051 core matrix: board metadata / lifecycle guards
# ---------------------------------------------------------------------------


def test_set_current_board_blocked_when_kanban_home_outside_root(
    isolated_home, live_root, monkeypatch
):
    """Switching the current-board symlink under an outside kanban home is refused."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.set_current_board("default")


def test_clear_current_board_blocked_when_kanban_home_outside_root(
    isolated_home, live_root, monkeypatch
):
    """Clearing the current-board symlink under an outside kanban home is refused."""
    kb.set_current_board("default")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.clear_current_board()


def test_create_board_blocked_when_kanban_home_outside_root(
    isolated_home, live_root, monkeypatch
):
    """Board creation under an outside root is refused before touching disk."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.create_board("outside")


def test_remove_board_blocked_when_kanban_home_outside_root(
    isolated_home, live_root, monkeypatch
):
    """Board removal under an outside root is refused before touching disk."""
    kb.create_board("toremove")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.remove_board("toremove")


# ---------------------------------------------------------------------------
# t_0645f051 core matrix: attachment / workspace guards
# ---------------------------------------------------------------------------


def test_store_attachment_bytes_blocked_when_attachments_root_outside(
    isolated_home, live_root, monkeypatch
):
    """Writing an attachment blob outside the declared temp root is refused."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="attachment host", assignee="reviewer")

    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT", str(live_root / "attachments")
    )

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.store_attachment_bytes(
                conn, tid, "note.txt", b"leaked", uploaded_by="test"
            )


def test_resolve_workspace_blocked_when_workspaces_root_outside(
    isolated_home, live_root, monkeypatch
):
    """Creating a scratch workspace directory outside the temp root is refused."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="workspace host", assignee="reviewer")

    monkeypatch.setenv(
        "HERMES_KANBAN_WORKSPACES_ROOT", str(live_root / "workspaces")
    )
    task = kb.get_task(conn, tid)
    assert task is not None

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.resolve_workspace(task)


# ---------------------------------------------------------------------------
# t_0645f051 core matrix: context lifecycle and reads
# ---------------------------------------------------------------------------


def test_isolation_resets_after_context_exit(isolated_home, live_root, monkeypatch):
    """Once the envelope closes, the same outside path is allowed again."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_root / "kanban.db"))

    with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
        with kb.kanban_mutation_isolation(isolated_home):
            kb.connect()

    # Outside the context there is no isolation root; connect() succeeds.
    kb.connect().close()


def test_nested_isolation_contexts_use_innermost_root(
    isolated_home, tmp_path, monkeypatch
):
    """A nested isolation context uses the innermost root."""
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    # Make nested_root look like the canonical default DB so connect() works.
    monkeypatch.setenv("HERMES_HOME", str(nested_root))

    # Outer envelope blocks nested_root because it is outside isolated_home.
    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.connect()

        # Inner envelope permits nested_root.
        with kb.kanban_mutation_isolation(nested_root):
            with kb.connect() as conn:
                tid = kb.create_task(conn, title="nested", assignee="reviewer")
            assert kb.get_task(conn, tid) is not None

    # After both exit, no isolation remains.
    monkeypatch.setenv("HERMES_HOME", str(isolated_home))
    kb.connect().close()


def test_reads_not_gated_by_mutation_isolation(
    isolated_home, live_root, monkeypatch
):
    """Read-only metadata reads do not require the path to be under the root."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        # Does not raise even though kanban_home points outside the root.
        meta = kb.read_board_metadata("default")

    assert meta["slug"] == "default"


# ---------------------------------------------------------------------------
# t_dae1e07e repair additions: write_txn validates actual DB identity
# ---------------------------------------------------------------------------


def test_write_txn_blocks_ambiguous_connection_identity(isolated_home, monkeypatch):
    """A connection whose main DB identity cannot be determined is refused."""
    import sqlite3

    with kb.kanban_mutation_isolation(isolated_home):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        with pytest.raises(PermissionError, match="could not determine the actual DB"):
            with kb.write_txn(conn):
                pass
        conn.close()


# ---------------------------------------------------------------------------
# t_dae1e07e repair additions: clear_board_maintenance guarded
# ---------------------------------------------------------------------------


def test_clear_board_maintenance_blocked_when_kanban_home_outside_root(
    isolated_home, live_root, monkeypatch
):
    """Clearing maintenance under an outside kanban home is refused."""
    kb.set_board_maintenance("default", enabled=True, reason="test")
    meta_path = kb.board_metadata_path("default")
    original_text = meta_path.read_text(encoding="utf-8")
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(live_root))

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="kanban_mutation_isolation"):
            kb.clear_board_maintenance("default")

    # File state unchanged: the temp-home board.json still has maintenance=True.
    assert meta_path.read_text(encoding="utf-8") == original_text
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert kb.board_in_maintenance(meta) is True


# ---------------------------------------------------------------------------
# t_dae1e07e repair additions: delete_attachment refuses outside-root path
# ---------------------------------------------------------------------------


def test_delete_attachment_blocked_for_outside_stored_path(
    isolated_home, live_root, monkeypatch
):
    """Deleting an attachment whose blob lives outside the root is refused."""
    outside_attachments = live_root / "attachments"
    outside_attachments.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", str(outside_attachments))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="attachment host", assignee="reviewer")
        att_id = kb.store_attachment_bytes(
            conn, tid, "note.txt", b"outside", uploaded_by="test"
        )
        att = kb.get_attachment(conn, att_id)
        assert att is not None
        stored = att.stored_path

    assert Path(stored).is_file()

    with kb.kanban_mutation_isolation(isolated_home):
        with kb.connect() as conn:
            with pytest.raises(PermissionError, match="attachment stored path"):
                kb.delete_attachment(conn, att_id)

    # DB row and file must remain intact.
    with kb.connect() as conn:
        assert kb.get_attachment(conn, att_id) is not None
    assert Path(stored).is_file()


# ---------------------------------------------------------------------------
# t_dae1e07e repair additions: worktree resolution refuses outside-root
# ---------------------------------------------------------------------------


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(path), "init", "--quiet"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test User"],
        check=True,
        capture_output=True,
    )
    # Worktree add needs a non-empty history so HEAD is a valid ref.
    readme = path / "README.md"
    readme.write_text("init\n")
    subprocess.run(
        ["git", "-C", str(path), "add", str(readme)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "init", "--quiet"],
        check=True,
        capture_output=True,
    )


def test_resolve_worktree_blocked_when_default_workdir_outside_root(
    isolated_home, live_root, monkeypatch
):
    """A worktree anchored on a repo outside the temp root is refused."""
    outside_repo = live_root / "outside_repo"
    outside_repo.mkdir()
    _git_init(outside_repo)

    kb.write_board_metadata("default", default_workdir=str(outside_repo))

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="worktree host",
            assignee="reviewer",
            workspace_kind="worktree",
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="workspace path .* resolves outside"):
            kb.resolve_workspace(task)


def test_resolve_worktree_blocked_when_explicit_path_outside_root(
    isolated_home, live_root, monkeypatch
):
    """An explicit worktree path outside the temp root is refused."""
    inside_repo = isolated_home / "inside_repo"
    inside_repo.mkdir()
    _git_init(inside_repo)

    outside_repo = live_root / "outside_repo"
    outside_repo.mkdir()
    _git_init(outside_repo)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="worktree host",
            assignee="reviewer",
            workspace_kind="worktree",
            workspace_path=str(outside_repo),
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    with kb.kanban_mutation_isolation(isolated_home):
        with pytest.raises(PermissionError, match="workspace path .* resolves outside"):
            kb.resolve_workspace(task)


# ---------------------------------------------------------------------------
# In-root behavior unaffected
# ---------------------------------------------------------------------------


def test_worktree_allowed_under_isolation_root(isolated_home):
    """A worktree under the isolation root materializes normally."""
    inside_repo = isolated_home / "inside_repo"
    inside_repo.mkdir()
    _git_init(inside_repo)
    kb.write_board_metadata("default", default_workdir=str(inside_repo))

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="worktree host",
            assignee="reviewer",
            workspace_kind="worktree",
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    with kb.kanban_mutation_isolation(isolated_home):
        p = kb.resolve_workspace(task)

    assert p.exists()
    assert "inside_repo" in str(p)
    assert tid in str(p)
