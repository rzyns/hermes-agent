"""Shared helper for first-class Attention Intake link-drop tasks.

Provides the contract-layer between CLI, dashboard, and worker tools
so that all three surfaces use the same body template, idempotency
algorithm, and register-write semantics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlunparse, urlparse

from hermes_cli import kanban_db as kb
from hermes_constants import get_default_hermes_root

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "attention-intake"
DEFAULT_ASSIGNEE = "link-analyst"
DEFAULT_TRIAGE = True
DEFAULT_PRIORITY = 0


def _artifact_root() -> Path:
    """Resolve the attention-intake artifact root from the active Hermes home."""
    return get_default_hermes_root() / "artifacts" / "attention-intake"

# ---------------------------------------------------------------------------
# URL canonicalisation
# ---------------------------------------------------------------------------


def canonical_url_hash(url: str) -> str:
    """Return SHA-256 of parsed-normalised URL as hex.

    Normalisation policy:
    - strip surrounding whitespace
    - normalize empty/root path to an empty path
    - lowercase scheme and host (netloc)
    - drop default ports (:80 for http, :443 for https)
    - preserve non-root path trailing slashes, path case, query order,
      fragment, and percent-encoding
    - drop empty fragment/hash

    This preserves distinct resources like ``/foo/`` vs ``/foo``,
    ``/Foo`` vs ``/foo``, and keeps percent-encoding round-trips stable.
    """
    url = url.strip()
    parts = urlparse(url)

    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()

    # Drop default ports
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Normalize only empty/root paths; preserve non-root trailing slashes.
    path = "" if parts.path in ("", "/") else parts.path

    # Drop empty fragment
    fragment = parts.fragment if parts.fragment else ""

    # Reassemble; leave query untouched (order preserved)
    normalised = urlunparse((scheme, netloc, path, "", parts.query, fragment))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def build_intake_link_body(
    url: str,
    context: Optional[str],
    note: Optional[str],
    source: str,
    board: str,
    assignee: str,
    idempotency_key: str,
    workspace_path: str,
) -> str:
    """Return the standard Markdown body for an intake-link task."""
    ctx_display = context.strip() if context else None
    note_display = note.strip() if note else None

    artifact_root = _artifact_root()
    return f"""## Link
{url}

### Context from dropper
{ctx_display or "(none provided)"}

### Operator note
{note_display or "(none)"}

### Register contract (auto-generated)
- **Status**: needs_assessment
- **Board**: {board}
- **Assignee**: {assignee}
- **Source**: {source}
- **Deduplication key**: {idempotency_key}
- **Artifact path**: {workspace_path}

---
> This task was created via the Attention Intake link-drop path.
> Worker must write/update the register at:
>   {artifact_root / "register.md"}
>   {artifact_root / "register.jsonl"}
"""


# ---------------------------------------------------------------------------
# Title helper
# ---------------------------------------------------------------------------


def _make_title(url: str) -> str:
    """Derive a concise title from URL, capped at 140 chars."""
    # Drop scheme for brevity if netloc is present.
    if "://" in url:
        _, _, remainder = url.partition("://")
    else:
        remainder = url
    display = remainder.strip("/")
    if len(display) > 140:
        display = display[:137] + "..."
    if not display:
        display = url
    return f"Link drop: {display}"


# ---------------------------------------------------------------------------
# Register writer (provisional)
# ---------------------------------------------------------------------------


def _write_register_entry(
    task_id: str,
    url: str,
    idempotency_key: str,
    board: str,
) -> None:
    """Append a minimal provisional entry to the JSONL register.

    Best-effort: if the directory isn't writable, log a warning and move on.
    The audit/health card (HP-AILD-03) will surface orphaned task ids.
    """
    jsonl_path = _artifact_root() / "register.jsonl"
    entry = {
        "event": "intake_link_created",
        "task_id": task_id,
        "url": url,
        "idempotency_key": idempotency_key,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board": board,
        "status": "needs_assessment",
    }
    try:
        _artifact_root().mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("Could not write provisional register entry: %s", exc)


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def create_intake_link(
    conn,
    *,
    url: str,
    context: Optional[str] = None,
    note: Optional[str] = None,
    board: str = DEFAULT_BOARD,
    assignee: str = DEFAULT_ASSIGNEE,
    triage: bool = DEFAULT_TRIAGE,
    priority: int = DEFAULT_PRIORITY,
    skills: Optional[Iterable[str]] = None,
    max_runtime_seconds: Optional[int] = None,
    idempotency_key: Optional[str] = None,
    source: str = "cli",
) -> str:
    """Create a task with the intake-link contract.

    Returns the task id. If ``idempotency_key`` is not supplied, a
    canonical URL hash is used as the default.

    After successful creation, this helper attempts:
    1. Mkdir the per-task artifact directory.
    2. Write a provisional JSONL register entry.
    3. Update the task row with the resolved ``workspace_path``.
    """
    if not url or not url.strip():
        raise ValueError("url is required")

    effective_idempotency = idempotency_key or canonical_url_hash(url)

    existing_row = None
    if effective_idempotency:
        existing_row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (effective_idempotency,),
        ).fetchone()
        if existing_row:
            return existing_row["id"]

    # Generate the task id first (idempotency will reuse existing).
    # We need the id to interpolate the workspace path.
    # But create_task returns the id and already checks idempotency.
    # So we pass workspace_path=None here, then patch it after.
    title = _make_title(url)
    
    new_task_id = kb.create_task(
        conn,
        title=title,
        body=None,  # Will patch after we know workspace_path
        assignee=assignee,
        created_by=source,
        workspace_kind="dir",
        workspace_path=None,
        tenant=None,
        priority=priority,
        parents=(),
        triage=triage,
        idempotency_key=effective_idempotency,
        max_runtime_seconds=max_runtime_seconds,
        skills=skills,
        initial_status="running",
    )

    # If this was an idempotency hit, create_task returned the existing id
    # and we must NOT overwrite the body or workspace_path.
    existing = kb.get_task(conn, new_task_id)
    # Robustly detect a truly-fresh row, even when the board has a
    # default_workdir that auto-fills workspace_path.  An idempotency-hit
    # will already carry our contract body; anything else needs patching.
    needs_patch = existing and (existing.body is None or "Attention Intake link-drop path." not in existing.body)
    if needs_patch:
        # Newly created by us in this call (fresh row). Patch ws_path + body.
        task_artifact_dir = _artifact_root() / new_task_id
        ws_path = str(task_artifact_dir)

        # Attempt mkdir; warn but don't fail.
        try:
            task_artifact_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Could not mkdir artifact dir %s: %s", task_artifact_dir, exc)

        # Build body with resolved paths.
        body = build_intake_link_body(
            url=url,
            context=context,
            note=note,
            source=source,
            board=board,
            assignee=assignee,
            idempotency_key=effective_idempotency,
            workspace_path=ws_path,
        )

        # Patch workspace_path and body on the row.
        kb.set_workspace_path(conn, new_task_id, ws_path)
        kb.update_task_body(conn, new_task_id, body)

        # Provisional register write.
        _write_register_entry(new_task_id, url, effective_idempotency, board)
    else:
        # Idempotency hit: row already has body/workspace_path from first call
        pass

    return new_task_id


# ---------------------------------------------------------------------------
# Convenience: update task body (not exposed by kanban_db as standalone)
# ---------------------------------------------------------------------------

# NOTE: We rely on a small addition to kanban_db for updating the task body.
# If `update_task_body` doesn't exist there, we'll add it below.
