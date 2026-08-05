"""Shared validation kernel for Kanban request payloads.

This module is the single source of truth for rules that previously lived in
three inconsistent copies (REST dashboard plugin, CLI argparse flags, and
``kanban_db``).  Follow-up adoption work will wire the three surfaces into the
kernel; this card only defines and tests the kernel itself.

Design principles
-----------------

- **Loud at the server**: every request body base uses ``extra="forbid"`` so
  unknown keys produce a typed error that names the offending key, rather than
  a silent 200 that discards the data.
- **Three request states for every optional field**: ``key absent``,
  ``key present with a value``, and ``key present with JSON ``null`` are all
  distinguishable.  We use a sentinel default + ``model_fields_set`` to detect
  presence; fields that need to express "clear this" provide a sibling
  ``clear_*`` boolean, while fields that simply omitted keep their current
  value.  This fixes the PATCH bug where ``{"workspace_path": null}`` was
  indistinguishable from an omitted key and therefore silently retained the
  old value.
- **One workspace/branch validator**: ``validate_workspace_spec`` encodes the
  allowed ``workspace_kind`` set, when ``workspace_path`` is required,
  absolute-vs-relative path handling, and ``branch_name`` restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Shared constants (mirrors hermes_cli.kanban_db.VALID_WORKSPACE_KINDS so the
# validator can be used without importing the heavy kanban_db module at schema
# import time, but kept in sync by a test).
# ---------------------------------------------------------------------------

VALID_WORKSPACE_KINDS: frozenset[str] = frozenset({"scratch", "worktree", "dir"})
REQUIRED_PATH_KINDS: frozenset[str] = frozenset({"dir", "worktree"})


# ---------------------------------------------------------------------------
# Strict payload base
# ---------------------------------------------------------------------------

class Undefined:
    """Sentinel meaning "this key was not supplied in the request".

    Pydantic's ``model_fields_set`` already distinguishes absent vs. present,
    but using an explicit sentinel as the field default makes the contract
    visible to callers and lets us write plain ``field is UNDEFINED`` checks in
    endpoint code without re-parsing the raw JSON.
    """

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False


UNDEFINED: Any = Undefined()


class StrictPayloadBase(BaseModel):
    """Base for all Kanban API request bodies.

    - ``extra="forbid"`` raises a typed 422 naming any unknown key.
    - ``populate_by_name=True`` lets callers use either camelCase or snake_case
      aliases where a model declares them, but models in this kernel stay
      snake_case to match the DB/CLI conventions.
    - ``str_strip_whitespace=True`` trims leading/trailing whitespace on all
      string fields so a field composed of whitespace normalises to the same
      value as an omitted field.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    def is_supplied(self, name: str) -> bool:
        """True if ``name`` was a key in the incoming request (even if null).

        This lets PATCH endpoints distinguish "explicit JSON null" from
        "omitted" without inspecting the raw bytes.  Combine with the field's
        own nullability to decide whether to clear or retain the existing DB
        value.
        """
        return name in self.model_fields_set


class OptionalUnset(StrictPayloadBase):
    """Mixin that gives a model an ``unset`` helper for optional fields.

    Usage: a field is declared as ``Optional[T] = UNDEFINED`` (not ``None``).
    Callers then use ``m.value_or_none("field_name")`` to get ``Optional[T]``:
    - absent or whitespace-only  -> ``None`` (caller may keep DB value)
    - present with JSON null     -> ``None`` and ``is_supplied`` is True
    - present with a value         -> the value
    """

    def value_or_none(self, name: str) -> Any:
        """Return the value of ``name`` if supplied and not UNDEFINED, else None."""
        if not self.is_supplied(name):
            return None
        value = getattr(self, name)
        if isinstance(value, Undefined):
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value


# ---------------------------------------------------------------------------
# Workspace/branch validator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkspaceSpec:
    """Normalised workspace specification ready for storage."""

    workspace_kind: Literal["scratch", "worktree", "dir"]
    workspace_path: Optional[str]
    branch_name: Optional[str]


def _normalise_branch_name(value: Optional[str]) -> Optional[str]:
    """Return the branch name with outer whitespace removed, or None if empty.

    Normalisation only trims ASCII whitespace and line terminators at the
    boundaries.  It never removes internal whitespace, and it never turns an
    invalid name into a valid one: a name that is empty after trimming stays
    ``None``, and a name with internal whitespace or a leading ``-`` is still
    invalid (the caller must enforce those rules separately).
    """
    if value is None:
        return None
    trimmed = value.strip(" \t\r\n")
    if not trimmed:
        return None
    return trimmed


def validate_workspace_spec(
    workspace_kind: str,
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    *,
    require_path_for: frozenset[str] = REQUIRED_PATH_KINDS,
) -> WorkspaceSpec:
    """Validate and normalise a workspace/branch triple.

    Rules (chosen deliberately because the three legacy copies disagreed):

    1. ``workspace_kind`` must be one of ``scratch``, ``worktree``, ``dir``.
    2. ``workspace_path`` is required for ``dir`` and ``worktree`` by default.
       Callers that have a later bounded resolver (for example
       ``kanban_db.create_task`` deriving ``<repo>/.worktrees/<task-id>`` from
       a board ``default_workdir`` or project link) may pass
       ``require_path_for=frozenset()`` to defer the path requirement.  A
       supplied path is still validated (absolute, non-empty).  ``scratch``
       normally stores ``None``; an explicit scratch path is validated as
       absolute.
    3. Supplied paths must be absolute after ``~`` expansion.  Relative paths
       are rejected because they are ambiguous against the dispatcher's CWD
       and are a confused-deputy traversal risk.
    4. ``branch_name`` is permitted only when ``workspace_kind == "worktree"``.
    5. Branch-name normalisation strips outer whitespace.  A name that is
       empty after trimming becomes ``None``.  Internal whitespace and a
       leading ``-`` are rejected here so every surface shares the same
       stricter branch syntax.

    Returns a :class:`WorkspaceSpec` with the normalised values.  The path is
    returned as the expanded string so CLI/REST callers and tests can compare
    it directly.
    """
    kind = str(workspace_kind).strip().lower()
    if kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )

    normalised_path: Optional[str] = None
    if kind == "scratch":
        # Scratch normally stores no path (the dispatcher creates and cleans a
        # per-board managed directory).  A supplied path is a legacy/degraded
        # case that the DB still accepts; if given, it must be absolute so
        # resolve_workspace can apply the same guard as dir.
        if workspace_path is not None and not isinstance(workspace_path, Undefined):
            if str(workspace_path).strip():
                normalised_path = _validate_absolute_path(workspace_path)
    elif kind == "dir":
        if workspace_path is None or isinstance(workspace_path, Undefined):
            if "dir" in require_path_for:
                raise ValueError(f"workspace_kind={kind!r} requires workspace_path")
            normalised_path = None
        else:
            normalised_path = _validate_absolute_path(workspace_path)
    elif kind == "worktree":
        if workspace_path is None or isinstance(workspace_path, Undefined):
            if "worktree" in require_path_for:
                raise ValueError(f"workspace_kind={kind!r} requires workspace_path")
            normalised_path = None
        else:
            normalised_path = _validate_absolute_path(workspace_path)
    else:  # pragma: no cover - guarded above
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )

    normalised_branch = _normalise_branch_name(branch_name)
    if normalised_branch is not None:
        if kind != "worktree":
            raise ValueError("branch_name is only valid for worktree workspaces")
        if normalised_branch.startswith("-"):
            raise ValueError("branch_name must not start with '-'")
        if any(ch.isspace() for ch in normalised_branch):
            raise ValueError("branch has internal whitespace")

    return WorkspaceSpec(
        workspace_kind=kind,  # type: ignore[arg-type]
        workspace_path=normalised_path,
        branch_name=normalised_branch,
    )


def _validate_absolute_path(value: Any) -> str:
    """Return ``value`` expanded and absolute, or raise a descriptive ValueError."""
    normalised = str(value).strip()
    if not normalised:
        raise ValueError("workspace_path must be non-empty")
    expanded = Path(normalised).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"workspace_path {value!r} must be absolute "
            f"(relative paths are ambiguous against the dispatcher's CWD)"
        )
    return str(expanded)


# ---------------------------------------------------------------------------
# Typed errors for REST surfaces
# ---------------------------------------------------------------------------

class PayloadValidationError(ValueError):
    """Validation failure that carries the offending key for 422 responses."""

    def __init__(self, message: str, *, key: Optional[str] = None) -> None:
        super().__init__(message)
        self.key = key
        self.message = message

    def __str__(self) -> str:
        if self.key:
            return f"{self.key}: {self.message}"
        return self.message
