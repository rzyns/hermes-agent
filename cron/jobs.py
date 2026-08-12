"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import shutil
import tempfile
import threading
import time
import os
import re
import uuid

# Cross-process advisory file locking for jobs.json critical sections.
# fcntl is Unix-only; on Windows fall back to msvcrt. Either may be absent,
# in which case _jobs_lock() degrades to in-process locking only (the old
# behaviour) rather than failing.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
from datetime import datetime, timedelta
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any, Set, Tuple, Union, Collection

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_replace, atomic_write_text

# ``croniter`` compiles ~15 ms of regexes at import and only matters for
# 5-field cron expressions. Resolve lazily; ``HAS_CRONITER`` stays a module
# attribute (tests monkeypatch it, and a monkeypatched value wins because
# ``_ensure_croniter`` only probes while it's still None).
croniter = None
HAS_CRONITER: Optional[bool] = None


def _ensure_croniter() -> bool:
    """Import croniter on first use; honor a pre-set HAS_CRONITER override."""
    global croniter, HAS_CRONITER
    if HAS_CRONITER is None:
        try:
            from croniter import croniter as _croniter
            croniter = _croniter
            HAS_CRONITER = True
        except ImportError:
            HAS_CRONITER = False
    return bool(HAS_CRONITER)

# =============================================================================
# Configuration
# =============================================================================

# Cron is per-profile by design (issue #4707). Each profile owns its own cron
# store under its own HERMES_HOME, and a profile-scoped gateway runs that
# profile's jobs under that same HERMES_HOME — so a job authored in profile
# `coder` lives in `~/.hermes/profiles/coder/cron/jobs.json` and executes with
# `coder`'s `.env`, `config.yaml`, and skills. We deliberately anchor on
# `get_hermes_home()` (the active profile home), NOT `get_default_hermes_root()`
# (the shared root). Anchoring at the root would funnel every profile's jobs
# into one shared `jobs.json` and run them under whatever HERMES_HOME the
# ticker process happens to have — leaking config/credentials/skills across
# profiles (the security boundary #4707 was filed for). Do NOT change this to
# the default root: that re-breaks per-profile isolation. See also the dynamic
# `_get_hermes_home()` / `_get_lock_paths()` resolution in cron/scheduler.py.
HERMES_DIR = get_hermes_home().resolve()
# These constants remain the default-profile fallback and a compatibility
# surface for existing callers/tests. Cross-profile callers must scope paths
# with use_cron_store() instead of mutating them process-wide.
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
# Heartbeat file the in-process ticker touches on every loop iteration. The
# gateway process and the (separate) ``hermes cron status`` process share it
# so status can tell whether the ticker THREAD is alive, not just whether the
# gateway PROCESS exists — a ticker that dies silently inside a live gateway
# would otherwise report healthy (#32612, #32895).
TICKER_HEARTBEAT_FILE = CRON_DIR / "ticker_heartbeat"
# Last tick that completed WITHOUT raising. Distinguishing this from the plain
# heartbeat lets status detect a ticker that is alive but failing every tick.
TICKER_SUCCESS_FILE = CRON_DIR / "ticker_last_success"
# Default ticker loop interval (seconds). The single source of truth shared by
# the in-process ticker (cron/scheduler_provider.py) and the staleness
# threshold in `hermes cron status` (hermes_cli/cron.py), so the two never
# drift apart.
TICKER_INTERVAL_SECONDS = 60

# In-process lock protecting load_jobs→modify→save_jobs cycles.
# Required when tick() runs jobs in parallel threads — without this,
# concurrent mark_job_run / advance_next_run calls can clobber each other.
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()

# Upper bound on waiting for the cross-process .jobs.lock flock (#60703).
# Every cron function in the process funnels through _jobs_lock(), and the
# flock is taken while holding the process-wide RLock — so an unbounded wait
# on a lock held by a wedged sibling process silently freezes the ticker
# heartbeat and every job forever.  30s is orders of magnitude above any
# legitimate critical section (field updates only) while keeping the ticker's
# worst-case stall well under one status-alarm threshold.
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0
OUTPUT_DIR = CRON_DIR / "output"
ONESHOT_GRACE_SECONDS = 120


def rewrite_skill_refs(
    consolidated: Optional[Dict[str, str]] = None,
    pruned: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite cron job skill references after a curator consolidation pass.

    When the curator consolidates a skill X into umbrella Y (or archives X
    as pruned), any cron job that lists ``X`` in its ``skills`` field will
    fail to load ``X`` at run time — the scheduler logs a warning and
    skips the skill, so the job runs without the instructions it was
    scheduled to follow. See cron/scheduler.py where ``skill_view`` is
    called per skill name.

    This function repairs cron jobs in-place:

    - A skill listed in ``consolidated`` is replaced with its umbrella
      target (the ``into`` value). If the umbrella is already in the
      job's skill list, the stale name is dropped without duplication.
    - A skill listed in ``pruned`` is dropped outright — there is no
      forwarding target.
    - Ordering and other skills in the list are preserved.
    - The legacy ``skill`` field is realigned via ``_apply_skill_fields``.

    Args:
        consolidated: mapping of ``old_skill_name -> umbrella_skill_name``.
        pruned: list of skill names that were archived with no forwarding
            target.

    Returns a report dict::

        {
            "rewrites": [
                {
                    "job_id": ...,
                    "job_name": ...,
                    "before": [...],
                    "after": [...],
                    "mapped": {"old": "new", ...},
                    "dropped": ["old", ...],
                },
                ...
            ],
            "jobs_updated": N,
            "jobs_scanned": M,
        }

    Best-effort: exceptions from loading/saving propagate to the caller so
    tests can assert behaviour; the curator invocation site wraps this
    call in a try/except so a failure here never breaks the curator.
    """
    consolidated = dict(consolidated or {})
    pruned_set = set(pruned or [])
    # A skill listed in both wins as "consolidated" — it has a target,
    # which is the more useful of the two outcomes.
    pruned_set -= set(consolidated.keys())

    if not consolidated and not pruned_set:
        return {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    with _jobs_lock():
        jobs = load_jobs()
        rewrites: List[Dict[str, Any]] = []
        changed = False

        for job in jobs:
            skills_before = _normalize_skill_list(job.get("skill"), job.get("skills"))
            if not skills_before:
                continue

            mapped: Dict[str, str] = {}
            dropped: List[str] = []
            new_skills: List[str] = []

            for name in skills_before:
                if name in consolidated:
                    target = consolidated[name]
                    mapped[name] = target
                    if target and target not in new_skills:
                        new_skills.append(target)
                elif name in pruned_set:
                    dropped.append(name)
                elif name not in new_skills:
                    new_skills.append(name)

            if not mapped and not dropped:
                continue

            job["skills"] = new_skills
            job["skill"] = new_skills[0] if new_skills else None
            changed = True

            rewrites.append({
                "job_id": job.get("id"),
                "job_name": job.get("name") or job.get("id"),
                "before": list(skills_before),
                "after": list(new_skills),
                "mapped": mapped,
                "dropped": dropped,
            })

        if changed:
            save_jobs(jobs)
            logger.info(
                "Curator rewrote skill references in %d cron job(s)", len(rewrites)
            )

        return {
            "rewrites": rewrites,
            "jobs_updated": len(rewrites),
            "jobs_scanned": len(jobs),
        }
