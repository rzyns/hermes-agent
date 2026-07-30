import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hermes_cli import active_sessions


def test_resolve_max_concurrent_sessions_values(caplog):
    assert active_sessions.resolve_max_concurrent_sessions({}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": None}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": 0}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": -1}) is None
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": "3"}) == 3
    assert (
        active_sessions.resolve_max_concurrent_sessions(
            {"gateway": {"max_concurrent_sessions": 4}}
        )
        == 4
    )
    assert (
        active_sessions.resolve_max_concurrent_sessions(
            {"max_concurrent_sessions": 2, "gateway": {"max_concurrent_sessions": 4}}
        )
        == 2
    )

    caplog.set_level(logging.WARNING)
    assert active_sessions.resolve_max_concurrent_sessions({"max_concurrent_sessions": "many"}) is None
    assert any(
        "Ignoring invalid max_concurrent_sessions='many'" in record.message
        for record in caplog.records
    )


def test_live_lease_survives_epoch_create_time_drift(tmp_path, monkeypatch):
    """WSL clock drift must not make a live lease look PID-recycled."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    next_epoch = [100.0]

    class _DriftingProcess:
        def __init__(self, _pid):
            pass

        def create_time(self):
            value = next_epoch[0]
            next_epoch[0] += 3.0
            return value

    monkeypatch.setattr("psutil.Process", _DriftingProcess)
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr(
        "gateway.status.get_process_start_time", lambda _pid: 4242
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="first",
        surface="cli",
        config={"max_concurrent_sessions": 1},
    )
    assert lease is not None
    assert message is None

    blocked_lease = None
    try:
        blocked_lease, blocked_message = active_sessions.try_acquire_active_session(
            session_id="second",
            surface="cli",
            config={"max_concurrent_sessions": 1},
        )
        assert blocked_lease is None
        assert blocked_message is not None
    finally:
        if blocked_lease is not None:
            blocked_lease.release()
        lease.release()


def test_new_lease_is_safe_for_legacy_reader(tmp_path, monkeypatch):
    """Old peers must not interpret the stable fingerprint as epoch seconds."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr(
        "gateway.status.get_process_start_time", lambda _pid: 4242
    )

    lease, message = active_sessions.try_acquire_active_session(
        session_id="mixed-version",
        surface="cli",
        config={"max_concurrent_sessions": 1},
    )
    assert lease is not None
    assert message is None
    try:
        [stored] = active_sessions.active_session_registry_snapshot()
        # Pre-stable-fingerprint Hermes treats a null legacy value as PID-only.
        assert stored["process_start_time"] is None
        assert stored["stable_process_start_time"] == 4242
        assert (
            stored["process_start_time_source"]
            == active_sessions._PROCESS_START_TIME_SOURCE
        )
    finally:
        lease.release()


def test_pid_reuse_guard_only_compares_stable_fingerprints(monkeypatch):
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr(active_sessions, "_process_start_time", lambda _pid: 200)

    assert not active_sessions._pid_alive(
        123,
        stable_process_start_time=100,
        process_start_time_source=active_sessions._PROCESS_START_TIME_SOURCE,
    )
    assert active_sessions._pid_alive(123, stable_process_start_time=100)


def test_pid_reuse_guard_retains_live_pid_without_comparable_fingerprint(
    monkeypatch,
):
    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr(active_sessions, "_process_start_time", lambda _pid: 200)

    stable_source = active_sessions._PROCESS_START_TIME_SOURCE
    assert active_sessions._pid_alive(
        123,
        stable_process_start_time=None,
        process_start_time_source=stable_source,
    )
    assert active_sessions._pid_alive(
        123,
        stable_process_start_time="malformed",
        process_start_time_source=stable_source,
    )
    assert active_sessions._pid_alive(
        123,
        stable_process_start_time=100,
        process_start_time_source="future-version",
    )

    monkeypatch.setattr(active_sessions, "_process_start_time", lambda _pid: None)
    assert active_sessions._pid_alive(
        123,
        stable_process_start_time=100,
        process_start_time_source=stable_source,
    )


def test_cross_process_acquire_claims_only_one_last_slot(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo_root = Path(__file__).resolve().parents[2]
    ready_dir = tmp_path / "ready"
    ready_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    go_file = tmp_path / "go"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo_root)
    script = (
        "import os, time\n"
        "from pathlib import Path\n"
        "from hermes_cli.active_sessions import try_acquire_active_session\n"
        "idx = os.environ['WORKER_INDEX']\n"
        "worker_count = int(os.environ['WORKER_COUNT'])\n"
        "delayed_worker = os.environ.get('DELAYED_WORKER_INDEX')\n"
        "ready_dir = Path(os.environ['READY_DIR'])\n"
        "results_dir = Path(os.environ['RESULTS_DIR'])\n"
        "go_file = Path(os.environ['GO_FILE'])\n"
        "(ready_dir / idx).write_text('ready', encoding='utf-8')\n"
        "deadline = time.time() + 10\n"
        "while not go_file.exists():\n"
        "    if time.time() > deadline:\n"
        "        raise RuntimeError('timed out waiting for go file')\n"
        "    time.sleep(0.01)\n"
        "if idx == delayed_worker:\n"
        "    time.sleep(2.5)\n"
        "lease, message = try_acquire_active_session(\n"
        "    session_id=f'process-{idx}',\n"
        "    surface='cli',\n"
        "    config={'max_concurrent_sessions': 1},\n"
        ")\n"
        "if lease is None:\n"
        "    (results_dir / idx).write_text('BLOCK', encoding='utf-8')\n"
        "    print('BLOCK', flush=True)\n"
        "else:\n"
        "    (results_dir / idx).write_text('OK', encoding='utf-8')\n"
        "    print('OK', flush=True)\n"
        "    deadline = time.time() + 10\n"
        "    while len(list(results_dir.iterdir())) < worker_count:\n"
        "        if time.time() > deadline:\n"
        "            raise RuntimeError('timed out waiting for all workers to attempt acquire')\n"
        "        time.sleep(0.01)\n"
        "    lease.release()\n"
    )
    workers: list[subprocess.Popen[str]] = []
    try:
        for index in range(6):
            worker_env = env.copy()
            worker_env["WORKER_INDEX"] = str(index)
            worker_env["WORKER_COUNT"] = "6"
            worker_env["DELAYED_WORKER_INDEX"] = "5"
            worker_env["READY_DIR"] = str(ready_dir)
            worker_env["RESULTS_DIR"] = str(results_dir)
            worker_env["GO_FILE"] = str(go_file)
            workers.append(
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    env=worker_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )

        deadline = time.time() + 10
        while len(list(ready_dir.iterdir())) < len(workers):
            if time.time() > deadline:
                raise AssertionError("workers did not become ready")
            time.sleep(0.01)
        go_file.write_text("go", encoding="utf-8")

        outputs = []
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=10)
            assert worker.returncode == 0, stderr
            outputs.append(stdout.strip())
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.communicate()

    assert outputs.count("OK") == 1
    assert outputs.count("BLOCK") == len(workers) - 1
    assert active_sessions.active_session_registry_snapshot() == []




def test_release_orphaned_leases_reclaims_only_unowned_own_pid_entries(tmp_path, monkeypatch):
    """A long-lived server must reclaim leases whose session skipped teardown.

    ``_prune_dead`` only fires when the owning pid dies, so a ``hermes
    dashboard`` running for days holds a leaked lease until restart. The
    process reconciles against the leases it still owns instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cfg = {"max_concurrent_sessions": 5}
    kept, orphan = (
        active_sessions.try_acquire_active_session(
            session_id=sid, surface="desktop", config=cfg
        )[0]
        for sid in ("kept", "orphaned")
    )
    # Another live process's lease is not ours to reclaim.
    active_sessions._write_entries(
        active_sessions._state_path(),
        active_sessions._read_entries(active_sessions._state_path())
        + [{"lease_id": "elsewhere", "session_id": "other", "surface": "cli", "pid": os.getpid() }],
    )

    assert active_sessions.release_orphaned_leases({kept.lease_id, "elsewhere"}) == 1
    assert sorted(
        entry["session_id"]
        for entry in active_sessions.active_session_registry_snapshot()
    ) == ["kept", "other"]
    assert orphan is not None
