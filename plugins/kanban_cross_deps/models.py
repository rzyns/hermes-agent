"""Typed cross-board edge model for the kanban-cross-deps plugin.

Preserves explicit blocking semantics, provenanced source facts, and
required_parent_statuses so scheduler policy is data-driven, not kind-driven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Canonical edge kinds
# ---------------------------------------------------------------------------

VALID_EDGE_KINDS: frozenset[str] = frozenset(
    {
        "blocks",
        "depends_on",
        "depends_on_decision",
        "informed_by",
        "derived_from_research",
        "feeds",
        "supersedes",
        "related",
    }
)

DEFAULT_REQUIRED_PARENT_STATUSES: list[str] = ["done", "archived"]

# ---------------------------------------------------------------------------
# Edge dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossBoardEdge:
    """A canonical cross-board dependency edge.

    Fields mirror the durable-plan SQL schema and preserve provenance
    so downstream layers can distinguish canonical registry facts from
    inferred candidates.
    """

    id: str
    parent_board: str
    parent_id: str
    child_board: str
    child_id: str
    kind: str
    blocking: bool = True
    required_parent_statuses: list[str] = field(
        default_factory=lambda: list(DEFAULT_REQUIRED_PARENT_STATUSES)
    )
    source: str = "canonical"
    created_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.parent_board.strip():
            raise ValueError("parent_board is required")
        if not self.parent_id.strip():
            raise ValueError("parent_id is required")
        if not self.child_board.strip():
            raise ValueError("child_board is required")
        if not self.child_id.strip():
            raise ValueError("child_id is required")
        if self.kind not in VALID_EDGE_KINDS:
            raise ValueError(f"kind must be one of {VALID_EDGE_KINDS}, got {self.kind!r}")
        if not self.source.strip():
            raise ValueError("source is required")
        # Coerce mutable defaults
        object.__setattr__(self, "required_parent_statuses", list(self.required_parent_statuses))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_board": self.parent_board,
            "parent_id": self.parent_id,
            "child_board": self.child_board,
            "child_id": self.child_id,
            "kind": self.kind,
            "blocking": self.blocking,
            "required_parent_statuses": list(self.required_parent_statuses),
            "source": self.source,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CrossBoardEdge":
        """Build from a sqlite3.Row dict."""
        return cls(
            id=str(row["id"]),
            parent_board=str(row["parent_board"]),
            parent_id=str(row["parent_id"]),
            child_board=str(row["child_board"]),
            child_id=str(row["child_id"]),
            kind=str(row["kind"]),
            blocking=bool(row["blocking"]),
            required_parent_statuses=_parse_json_list(row["required_parent_statuses"]),
            source=str(row["source"]),
            created_by=row.get("created_by") or None,
            created_at=_parse_ts(row.get("created_at")),
            updated_at=_parse_ts(row.get("updated_at")),
            metadata=_parse_json_dict(row.get("metadata")),
        )

    def parent_key(self) -> tuple[str, str]:
        return (self.parent_board, self.parent_id)

    def child_key(self) -> tuple[str, str]:
        return (self.child_board, self.child_id)


# ---------------------------------------------------------------------------
# Helper parsers (defensive: accept JSON string, list, dict, or None)
# ---------------------------------------------------------------------------

def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_REQUIRED_PARENT_STATUSES)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        return list(DEFAULT_REQUIRED_PARENT_STATUSES)
    if isinstance(value, list):
        return [str(v) for v in value]
    return list(DEFAULT_REQUIRED_PARENT_STATUSES)


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, int):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
