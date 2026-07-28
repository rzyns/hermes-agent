"""Regression tests for the jobs.json stale-snapshot clobber (mechanism M2).

Background — the failure this guards against, observed 2026-07-28.

``_jobs_lock()`` (see ``test_jobs_crossprocess_lock.py``) closed mechanism M1:
two writers interleaving inside a load→modify→save window with no cross-process
lock. It establishes *mutual exclusion*.

It does not establish *freshness*. A process that called ``load_jobs()`` hours
ago and then calls ``save_jobs()`` takes the flock entirely legitimately, and
writes its stale in-memory list over everything that landed in between. The
lock is honoured; the data is still destroyed.

Observed cost: a gateway that had been running ~11 hours flushed its snapshot
over a cron job created minutes earlier by a CLI worker, and the job simply
ceased to exist. The scripts and artifacts that job's card produced survived —
only the cron registration vanished, which is the signature of a jobs.json-scoped
loss rather than a failed worker.

The fix records the store generation (``updated_at``) that a loaded list came
from, and refuses to write a list whose generation is no longer current.

Design constraint that shapes these tests: a *merge* on save is not viable.
Under the lock you can re-read the disk, but you cannot distinguish "the caller
deleted job X" from "the caller never knew about job X" — so merging would
resurrect deleted jobs and silently break ``cron remove``. Hence a guard that
refuses the write, plus ``test_explicit_deletion_still_works`` below to prove
the guard did not cost us deletion.
"""

import pytest

from cron import jobs


def _job(job_id: str, prompt: str = "hello") -> dict:
    return {"id": job_id, "name": f"job-{job_id}", "prompt": prompt, "enabled": True}


def _ids(job_list) -> set:
    return {j["id"] for j in job_list}


def test_stale_snapshot_does_not_clobber_newer_jobs(tmp_path):
    """The M2 failure, reduced.

    A long-lived reader loads, someone else adds a job, then the long-lived
    reader flushes its stale list. The newer job must survive.
    """
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([_job("a")])

        # A long-lived process (the gateway) loads once and holds the list.
        stale_snapshot = jobs.load_jobs()
        assert _ids(stale_snapshot) == {"a"}

        # Meanwhile a separate writer (a CLI invocation) registers a new job.
        current = jobs.load_jobs()
        current.append(_job("b"))
        jobs.save_jobs(current)
        assert _ids(jobs.load_jobs()) == {"a", "b"}

        # The long-lived process now flushes the snapshot it loaded earlier.
        # Pre-fix this silently deletes job "b".
        jobs.save_jobs(stale_snapshot)

        surviving = _ids(jobs.load_jobs())

    assert "b" in surviving, (
        "job 'b' was destroyed by a stale-snapshot write — mechanism M2. "
        f"surviving ids: {sorted(surviving)}"
    )
    assert surviving == {"a", "b"}


def test_explicit_deletion_still_works(tmp_path):
    """The guard must not resurrect deliberately removed jobs.

    This is the constraint that rules out merge-on-save, so it is the most
    important test in this file: if it ever fails, ``hermes cron remove`` is
    broken.
    """
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([_job("a"), _job("b")])

        current = jobs.load_jobs()
        remaining = [j for j in current if j["id"] != "b"]
        jobs.save_jobs(remaining)

        after = _ids(jobs.load_jobs())

    assert after == {"a"}, f"deletion of 'b' did not stick: {sorted(after)}"


def test_normal_load_modify_save_roundtrip_is_unaffected(tmp_path):
    """The overwhelmingly common path must behave exactly as before."""
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([_job("a")])

        current = jobs.load_jobs()
        current.append(_job("b"))
        jobs.save_jobs(current)

        current = jobs.load_jobs()
        for j in current:
            if j["id"] == "a":
                j["prompt"] = "edited"
        jobs.save_jobs(current)

        after = jobs.load_jobs()

    assert _ids(after) == {"a", "b"}
    assert [j for j in after if j["id"] == "a"][0]["prompt"] == "edited"


def test_fresh_list_without_provenance_is_still_writable(tmp_path):
    """A caller that builds a list from scratch has no generation to check.

    Many call sites (and most of the existing test suite) do exactly this. They
    must keep working — the guard fails OPEN for untagged lists rather than
    refusing writes it cannot reason about.
    """
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([_job("a")])
        jobs.save_jobs([_job("x"), _job("y")])  # plain list, no provenance
        after = _ids(jobs.load_jobs())

    assert after == {"x", "y"}


def test_reload_after_rejected_write_sees_current_state(tmp_path):
    """After a stale write is refused, the caller can recover by reloading."""
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([_job("a")])
        stale_snapshot = jobs.load_jobs()

        current = jobs.load_jobs()
        current.append(_job("b"))
        jobs.save_jobs(current)

        jobs.save_jobs(stale_snapshot)  # refused

        # A fresh load reflects reality, and a write derived from it succeeds.
        recovered = jobs.load_jobs()
        assert _ids(recovered) == {"a", "b"}
        recovered.append(_job("c"))
        jobs.save_jobs(recovered)

        after = _ids(jobs.load_jobs())

    assert after == {"a", "b", "c"}
