"""Dependency provider that bridges the cross-board edge registry to the core seam.

The provider queries canonical blocking edges from the plugin registry,
resolves parent statuses across board-local Kanban DBs, and returns
ExternalBlocker records so the core scheduler can enforce the hard claim
and promotion invariants.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_dependencies import ExternalBlocker

from plugins.kanban_cross_deps.models import CrossBoardEdge
from plugins.kanban_cross_deps.store import CrossBoardRegistry

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_PARENT_STATUSES: list[str] = ["done", "archived"]


def make_provider(registry: CrossBoardRegistry | None = None) -> CrossBoardDependencyProvider:
    """Factory for the plugin's default dependency provider."""
    return CrossBoardDependencyProvider(registry=registry)


class CrossBoardDependencyProvider:
    """KanbanDependencyProvider implementation backed by CrossBoardRegistry.

    Fail-closed semantics:
    - blocking edges with unsatisfied parents → unsatisfied ExternalBlocker
    - blocking edges with dangling parents → unsatisfied ExternalBlocker
    - non-blocking edges → ignored for scheduler (not returned)
    - any unexpected exception → bubbles up; core seam wraps as provider_error
    """

    name = "kanban_cross_deps"

    def __init__(self, registry: CrossBoardRegistry | None = None) -> None:
        self.registry = registry or CrossBoardRegistry()

    def blockers_for(
        self, *, board: str, task_id: str
    ) -> Iterable[ExternalBlocker | Mapping[str, Any]]:
        """Return unsatisfied external blockers for ``board/task_id``.

        Only blocking edges are considered.  Non-blocking edges are ignored
        so they do not participate in scheduler enforcement.
        """
        edges = self.registry.list_edges(child_board=board, child_id=task_id, blocking=True)
        blockers: list[ExternalBlocker] = []
        for edge in edges:
            blocker = self._resolve_edge(edge)
            if blocker and not blocker.satisfied:
                blockers.append(blocker)
        return blockers

    def _resolve_edge(self, edge: CrossBoardEdge) -> ExternalBlocker | None:
        """Resolve a single blocking edge into an ExternalBlocker."""
        required_statuses = edge.required_parent_statuses or DEFAULT_REQUIRED_PARENT_STATUSES

        try:
            parent_status = self._get_parent_status(edge.parent_board, edge.parent_id)
        except Exception as exc:
            # Dangling or unreadable parent board → fail-closed
            logger.warning(
                "Cross-board provider failed to resolve parent %s/%s for child %s/%s: %s",
                edge.parent_board, edge.parent_id, edge.child_board, edge.child_id, exc,
            )
            return ExternalBlocker(
                parent_board=edge.parent_board,
                parent_id=edge.parent_id,
                status=None,
                kind=edge.kind,
                reason=f"dangling or unreadable parent: {exc}",
                satisfied=False,
                provenance={
                    "edge_id": edge.id,
                    "required_parent_statuses": required_statuses,
                    "error_type": exc.__class__.__name__,
                },
            )

        if parent_status is None:
            # Parent task does not exist → fail-closed
            return ExternalBlocker(
                parent_board=edge.parent_board,
                parent_id=edge.parent_id,
                status=None,
                kind=edge.kind,
                reason="parent task not found",
                satisfied=False,
                provenance={
                    "edge_id": edge.id,
                    "required_parent_statuses": required_statuses,
                },
            )

        satisfied = parent_status in required_statuses
        return ExternalBlocker(
            parent_board=edge.parent_board,
            parent_id=edge.parent_id,
            status=parent_status,
            kind=edge.kind,
            reason=(
                f"parent status '{parent_status}' not in {required_statuses}"
                if not satisfied
                else f"parent status '{parent_status}' satisfies {required_statuses}"
            ),
            satisfied=satisfied,
            provenance={
                "edge_id": edge.id,
                "required_parent_statuses": required_statuses,
            },
        )

    def _get_parent_status(self, parent_board: str, parent_id: str) -> str | None:
        """Open the parent board's kanban DB and read the task status.

        Returns ``None`` if the task row is absent.  Raises on DB errors
        so the caller can fail-closed.
        """
        conn = kb.connect(board=parent_board)
        try:
            task = kb.get_task(conn, parent_id)
            return task.status if task else None
        finally:
            conn.close()
