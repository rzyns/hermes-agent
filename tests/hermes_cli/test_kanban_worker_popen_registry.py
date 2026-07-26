"""Regression tests for the dispatcher's retained Popen handle registry.

The bug: `_default_spawn()` returned only `proc.pid` and dropped the Popen
object. Python's subprocess cleanup could reap the forgotten child before
Hermes's `waitpid()` loop ran, so `_classify_worker_exit()` had no status
and reported ``unknown``.  This module exercises the registry directly.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

import hermes_cli.kanban_db as kb


def _reset_active_workers() -> None:
    """Clear global state so tests do not interfere with each other."""
    kb._active_worker_procs.clear()
    kb._recent_worker_exits.clear()


def test_poll_active_worker_returns_infra_exit_code():
    """A retained handle for a child that exits 78 yields the right raw status."""
    _reset_active_workers()
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.exit({kb.KANBAN_INFRA_EXIT_CODE})"]
    )
    kb._register_active_worker(proc)
    time.sleep(0.3)

    reaped = kb._poll_active_workers()
    assert len(reaped) == 1
    pid, raw_status = reaped[0]
    assert pid == proc.pid
    assert raw_status == kb.KANBAN_INFRA_EXIT_CODE << 8


def test_poll_active_worker_returns_rate_limit_exit_code():
    """A retained handle for a child that exits 75 yields the right raw status."""
    _reset_active_workers()
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.exit({kb.KANBAN_RATE_LIMIT_EXIT_CODE})"]
    )
    kb._register_active_worker(proc)
    time.sleep(0.3)

    reaped = kb._poll_active_workers()
    assert len(reaped) == 1
    pid, raw_status = reaped[0]
    assert pid == proc.pid
    assert raw_status == kb.KANBAN_RATE_LIMIT_EXIT_CODE << 8


def test_poll_active_worker_returns_clean_exit_status():
    """A retained handle for a child that exits 0 yields the right raw status."""
    _reset_active_workers()
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    kb._register_active_worker(proc)
    time.sleep(0.3)

    reaped = kb._poll_active_workers()
    assert len(reaped) == 1
    pid, raw_status = reaped[0]
    assert pid == proc.pid
    assert raw_status == 0


def test_reap_worker_zombies_uses_active_worker_registry_first():
    """reap_worker_zombies() pulls statuses from retained handles."""
    _reset_active_workers()
    proc = subprocess.Popen(
        [sys.executable, "-c", f"import sys; sys.exit({kb.KANBAN_INFRA_EXIT_CODE})"]
    )
    kb._register_active_worker(proc)
    time.sleep(0.3)

    reaped = kb.reap_worker_zombies()
    assert proc.pid in reaped
    assert kb._classify_worker_exit(proc.pid) == ("infra_blocked", kb.KANBAN_INFRA_EXIT_CODE)
    assert proc.pid not in kb._active_worker_procs


def test_poll_active_workers_leaves_running_children_in_registry():
    """A still-running child is not reaped and remains tracked."""
    _reset_active_workers()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    kb._register_active_worker(proc)

    try:
        reaped = kb._poll_active_workers()
        assert reaped == []
        assert proc.pid in kb._active_worker_procs
        # The handle is still open so the child's stdout/stderr FDs stay alive.
        assert kb._active_worker_procs[proc.pid].returncode is None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        kb._close_active_worker(proc.pid)


def test_active_workers_age_trim_evicts_old_handles():
    """Very old retained handles are evicted even if poll never completed."""
    _reset_active_workers()

    class FakeProc:
        def __init__(self):
            self.pid = 99999
            self.returncode = None
            self._spawned_at = time.time() - kb._ACTIVE_WORKER_MAX_AGE_SECONDS - 1

    proc = FakeProc()
    kb._active_worker_procs[proc.pid] = proc  # type: ignore[assignment]

    kb._active_workers_age_trim()
    assert proc.pid not in kb._active_worker_procs


def test_poll_active_workers_evicts_completed_handles():
    """Completed handles are removed and closed."""
    _reset_active_workers()
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    kb._register_active_worker(proc)
    time.sleep(0.3)

    kb._poll_active_workers()
    assert proc.pid not in kb._active_worker_procs
    # The underlying process has been waited, so returncode is available on
    # the original object even after we evicted the registry entry.
    assert proc.returncode == 0
