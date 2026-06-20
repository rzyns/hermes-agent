"""Durable artifact manifest helpers for Kanban workers.

Workers often produce non-code deliverables (reports, screenshots, generated
media, spreadsheets) inside their task workspace. Free-form prose references are
easy to lose, so completion metadata should include a small manifest with
absolute paths, workspace-relative paths, existence checks, and containment
status. These helpers are advisory/building blocks; they do not block task
completion by themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FILENAME = "artifact-manifest.json"


def _resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _relative_to_or_none(path: Path, root: Path) -> str | None:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def artifact_manifest_path(workspace_path: Path | str) -> Path:
    """Return the canonical manifest path for a task workspace."""
    return _resolve_path(workspace_path) / MANIFEST_FILENAME


def build_artifact_manifest(
    *,
    task_id: str,
    workspace_path: Path | str,
    artifacts: Iterable[Path | str],
    board: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a serializable durable artifact manifest.

    ``artifacts`` should be absolute paths under ``workspace_path``. The helper
    records invalid/out-of-workspace paths instead of raising so callers can
    surface actionable diagnostics without hiding the evidence.
    """
    task_id = str(task_id or "").strip()
    if not task_id:
        raise ValueError("task_id is required")
    workspace = _resolve_path(workspace_path)
    entries: list[dict[str, Any]] = []
    for raw in artifacts:
        raw_text = str(raw)
        path = _resolve_path(raw)
        rel = _relative_to_or_none(path, workspace)
        exists = path.exists()
        is_file = path.is_file() if exists else False
        entries.append({
            "path": str(path),
            "raw_path": raw_text,
            "relative_path": rel,
            "exists": exists,
            "is_file": is_file,
            "size_bytes": path.stat().st_size if exists and is_file else None,
            "in_workspace": rel is not None,
        })
    valid = all(item["exists"] and item["in_workspace"] for item in entries)
    manifest: dict[str, Any] = {
        "version": 1,
        "task_id": task_id,
        "board": board,
        "workspace_path": str(workspace),
        "manifest_path": str(artifact_manifest_path(workspace)),
        "valid": valid,
        "artifacts": entries,
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    return manifest


def write_artifact_manifest(
    *,
    task_id: str,
    workspace_path: Path | str,
    artifacts: Iterable[Path | str],
    board: str | None = None,
    metadata: dict[str, Any] | None = None,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build and write ``artifact-manifest.json`` under the workspace.

    Returns the manifest dict, including the final ``manifest_path``. Parent
    directories are created; artifact files themselves are never created or
    modified.
    """
    manifest = build_artifact_manifest(
        task_id=task_id,
        workspace_path=workspace_path,
        artifacts=artifacts,
        board=board,
        metadata=metadata,
    )
    out = _resolve_path(manifest_path) if manifest_path is not None else Path(manifest["manifest_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_path"] = str(out)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
