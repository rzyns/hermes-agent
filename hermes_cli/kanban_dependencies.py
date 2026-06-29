"""Cross-board dependency provider seam for the Kanban scheduler.

This module is intentionally small and storage-agnostic.  Core Kanban owns the
hard scheduling invariant (a task must not be claimed while unsatisfied
blockers exist), while plugins or future native storage can register providers
that report blockers outside the board-local ``task_links`` table.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class ExternalBlocker:
    """A dependency outside the current board-local ``task_links`` graph."""

    parent_board: str
    parent_id: str
    status: str | None = None
    kind: str = "blocks"
    reason: str = ""
    satisfied: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_board": self.parent_board,
            "parent_id": self.parent_id,
            "status": self.status,
            "kind": self.kind,
            "reason": self.reason,
            "satisfied": bool(self.satisfied),
            "provenance": dict(self.provenance or {}),
        }


class KanbanDependencyProvider(Protocol):
    """Provider contract for scheduler-visible external dependencies."""

    def blockers_for(self, *, board: str, task_id: str) -> Iterable[ExternalBlocker | Mapping[str, Any]]:
        """Return external blockers for ``board/task_id``.

        Providers should be local and deterministic. The scheduler may call this
        during claim/promotion paths, so implementations must avoid slow network
        calls and user interaction.
        """


_lock = threading.RLock()
_providers: list[KanbanDependencyProvider] = []


def register_dependency_provider(provider: KanbanDependencyProvider) -> KanbanDependencyProvider:
    """Register a dependency provider and return it as an unregister token."""

    if not hasattr(provider, "blockers_for"):
        raise TypeError("kanban dependency provider must define blockers_for()")
    with _lock:
        if provider not in _providers:
            _providers.append(provider)
    return provider


def unregister_dependency_provider(provider: KanbanDependencyProvider) -> None:
    """Unregister a previously registered provider if present."""

    with _lock:
        try:
            _providers.remove(provider)
        except ValueError:
            pass


@contextlib.contextmanager
def scoped_dependency_provider(provider: KanbanDependencyProvider):
    """Temporarily register ``provider`` for tests and short-lived callers."""

    token = register_dependency_provider(provider)
    try:
        yield token
    finally:
        unregister_dependency_provider(token)


def _provider_name(provider: KanbanDependencyProvider) -> str:
    return getattr(provider, "name", provider.__class__.__name__)


def _coerce_blocker(raw: ExternalBlocker | Mapping[str, Any]) -> ExternalBlocker:
    if isinstance(raw, ExternalBlocker):
        return raw
    if isinstance(raw, Mapping):
        return ExternalBlocker(
            parent_board=str(raw.get("parent_board") or raw.get("board") or ""),
            parent_id=str(raw.get("parent_id") or raw.get("task_id") or raw.get("id") or ""),
            status=raw.get("status"),
            kind=str(raw.get("kind") or "blocks"),
            reason=str(raw.get("reason") or ""),
            satisfied=bool(raw.get("satisfied", False)),
            provenance=raw.get("provenance") if isinstance(raw.get("provenance"), Mapping) else {},
        )
    raise TypeError(f"unsupported external blocker type: {type(raw).__name__}")


def blockers_for(*, board: str, task_id: str) -> list[ExternalBlocker]:
    """Return all provider-reported blockers for ``board/task_id``.

    Provider failures are represented as unsatisfied synthetic blockers so the
    scheduler fails closed when an enabled dependency provider cannot answer.
    """

    with _lock:
        providers = list(_providers)
    blockers: list[ExternalBlocker] = []
    for provider in providers:
        try:
            for raw in provider.blockers_for(board=board, task_id=task_id) or []:
                blockers.append(_coerce_blocker(raw))
        except Exception as exc:
            blockers.append(
                ExternalBlocker(
                    parent_board="<dependency-provider>",
                    parent_id=_provider_name(provider),
                    status=None,
                    kind="provider_error",
                    reason=str(exc),
                    satisfied=False,
                    provenance={
                        "provider": _provider_name(provider),
                        "error_type": exc.__class__.__name__,
                    },
                )
            )
    return blockers


def unsatisfied_blockers_for(*, board: str, task_id: str) -> list[ExternalBlocker]:
    """Return only blockers that should prevent scheduling."""

    return [blocker for blocker in blockers_for(board=board, task_id=task_id) if not blocker.satisfied]
