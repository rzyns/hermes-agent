"""Diagnostics for the kanban-cross-deps plugin.

Detects:
- dangling parent/child boards or tasks
- blocking cycles across local task_links and cross-board edges
- satisfied-upstream-but-child-stuck contradictions
- unsatisfied-upstream-but-child-ready/running contradictions
- provider failure / fail-closed evidence

Cycle detection treats each task as a node keyed by (board, task_id) and
walks both board-local ``task_links`` and canonical cross-board edges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from hermes_cli import kanban_db as kb

from plugins.kanban_cross_deps.models import CrossBoardEdge
from plugins.kanban_cross_deps.provider import (
    DEFAULT_REQUIRED_PARENT_STATUSES,
    CrossBoardDependencyProvider,
)
from plugins.kanban_cross_deps.store import CrossBoardRegistry

logger = logging.getLogger(__name__)

Node = tuple[str, str]  # (board, task_id)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CyclePath:
    path: list[Node]
    edges: list[dict[str, Any]]
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": [f"{b}/{t}" for b, t in self.path],
            "edges": list(self.edges),
            "blocking": self.blocking,
        }

    def normalized(self) -> tuple[tuple[str, str], ...]:
        """Rotate so the lexicographically smallest node is first."""
        nodes = self.path[:-1]  # drop closing duplicate
        if not nodes:
            return tuple()
        min_idx = min(range(len(nodes)), key=lambda i: nodes[i])
        rotated = nodes[min_idx:] + nodes[:min_idx]
        return tuple(rotated)


@dataclass(frozen=True)
class DanglingEdge:
    edge: CrossBoardEdge
    missing_side: str  # "parent" or "child"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge.id,
            "missing_side": self.missing_side,
            "reason": self.reason,
            "edge": self.edge.to_dict(),
        }


@dataclass(frozen=True)
class Contradiction:
    edge: CrossBoardEdge
    kind: str
    parent_status: str | None
    child_status: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge.id,
            "kind": self.kind,
            "parent_status": self.parent_status,
            "child_status": self.child_status,
            "reason": self.reason,
            "edge": self.edge.to_dict(),
        }


@dataclass(frozen=True)
class ProviderFailure:
    edge: CrossBoardEdge
    error_type: str
    error_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge.id,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Diagnostics engine
# ---------------------------------------------------------------------------

class CrossBoardDiagnostics:
    """Run cross-board dependency diagnostics."""

    def __init__(
        self,
        registry: CrossBoardRegistry | None = None,
        provider: CrossBoardDependencyProvider | None = None,
    ) -> None:
        self.registry = registry or CrossBoardRegistry()
        self.provider = provider or CrossBoardDependencyProvider(registry=self.registry)

    # -- public API ----------------------------------------------------------

    def run(
        self,
        _local_link_resolver: Callable[[str, str], list[str]] | None = None,
        task_filter: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run the full diagnostics suite and return a machine-readable report.

        If *task_filter* is provided as ``(board, task_id)``, only edges
        where that task participates are evaluated, and summary/counts are
        recomputed for the restricted scope.
        """
        if _local_link_resolver is None:
            _local_link_resolver = _default_local_children_resolver

        all_edges = self.registry.list_edges(limit=10000)
        if task_filter:
            _board, _tid = task_filter
            all_edges = [
                e for e in all_edges
                if (e.child_board == _board and e.child_id == _tid)
                or (e.parent_board == _board and e.parent_id == _tid)
            ]

        cycles = self._find_cycles(all_edges, _local_link_resolver)
        blocking_cycles = [c for c in cycles if c.blocking]
        info_cycles = [c for c in cycles if not c.blocking]

        dangling = self._find_dangling(all_edges)
        contradictions = self._find_contradictions(all_edges)
        provider_failures = self._find_provider_failures(all_edges)

        return {
            "cycles": {
                "blocking": [c.to_dict() for c in blocking_cycles],
                "informational": [c.to_dict() for c in info_cycles],
                "total": len(cycles),
            },
            "dangling": [d.to_dict() for d in dangling],
            "contradictions": [c.to_dict() for c in contradictions],
            "provider_failures": [p.to_dict() for p in provider_failures],
            "summary": {
                "blocking_cycles": len(blocking_cycles),
                "informational_cycles": len(info_cycles),
                "dangling": len(dangling),
                "contradictions": len(contradictions),
                "provider_failures": len(provider_failures),
            },
        }

    def would_create_cycle(
        self,
        parent_board: str,
        parent_id: str,
        child_board: str,
        child_id: str,
        blocking: bool = True,
        _local_link_resolver: Callable[[str, str], list[str]] | None = None,
    ) -> bool:
        """Return True if adding the given edge would create a cycle.

        Only blocking edges are rejected for cycles.  Non-blocking edges are
        allowed but will still be reported as informational cycles in
        :meth:`run`.
        """
        if not blocking:
            return False

        if _local_link_resolver is None:
            _local_link_resolver = _default_local_children_resolver

        all_edges = self.registry.list_edges(limit=10000)
        blocking_edges = [e for e in all_edges if e.blocking]

        start: Node = (child_board, child_id)
        target: Node = (parent_board, parent_id)

        if start == target:
            return True

        seen: set[Node] = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)

            board, task_id = node
            # Cross-board outgoing edges
            for edge in blocking_edges:
                if edge.parent_board == board and edge.parent_id == task_id:
                    dst: Node = (edge.child_board, edge.child_id)
                    if dst not in seen:
                        stack.append(dst)

            # Local outgoing edges
            for child_tid in _local_link_resolver(board, task_id):
                dst = (board, child_tid)
                if dst not in seen:
                    stack.append(dst)

        return False

    # -- internal cycle detection --------------------------------------------

    def _find_cycles(
        self,
        edges: list[CrossBoardEdge],
        local_link_resolver: Callable[[str, str], list[str]],
    ) -> list[CyclePath]:
        """Find all simple cycles in the combined local + cross-board graph."""
        # Build adjacency with metadata
        adj: dict[Node, list[tuple[Node, dict[str, Any]]]] = {}

        def _add_edge(src: Node, dst: Node, meta: dict[str, Any]) -> None:
            adj.setdefault(src, []).append((dst, meta))

        # Cross-board edges (blocking and non-blocking)
        for edge in edges:
            _add_edge(
                edge.parent_key(),
                edge.child_key(),
                {
                    "type": "cross_board",
                    "edge_id": edge.id,
                    "blocking": edge.blocking,
                    "kind": edge.kind,
                },
            )

        # Local edges from all boards mentioned in cross-board edges
        boards_involved: set[str] = set()
        for edge in edges:
            boards_involved.add(edge.parent_board)
            boards_involved.add(edge.child_board)
        # Always scan the current board so local-only cycles are detected too
        try:
            boards_involved.add(kb.get_current_board())
        except Exception:
            pass

        for board in boards_involved:
            try:
                conn = kb.connect(board=board)
                try:
                    rows = conn.execute(
                        "SELECT parent_id, child_id FROM task_links"
                    ).fetchall()
                    for r in rows:
                        src = (board, r["parent_id"])
                        dst = (board, r["child_id"])
                        _add_edge(
                            src,
                            dst,
                            {"type": "local", "blocking": True},
                        )
                finally:
                    conn.close()
            except Exception as exc:
                logger.warning(
                    "Could not read local links for board %s: %s", board, exc
                )

        # DFS cycle enumeration with deduplication
        cycles: list[CyclePath] = []
        seen_normalized: set[tuple[tuple[str, str], ...]] = set()

        for start_node in list(adj.keys()):
            self._dfs_cycles(
                node=start_node,
                adj=adj,
                stack_set=set(),
                stack_list=[],
                edge_stack=[],
                cycles=cycles,
                seen_normalized=seen_normalized,
            )

        return cycles

    def _dfs_cycles(
        self,
        node: Node,
        adj: dict[Node, list[tuple[Node, dict[str, Any]]]],
        stack_set: set[Node],
        stack_list: list[Node],
        edge_stack: list[dict[str, Any]],
        cycles: list[CyclePath],
        seen_normalized: set[tuple[tuple[str, str], ...]],
    ) -> None:
        if node in stack_set:
            idx = stack_list.index(node)
            cycle_nodes = stack_list[idx:] + [node]
            cycle_edges = edge_stack[idx:]
            cp = CyclePath(
                path=cycle_nodes,
                edges=cycle_edges,
                blocking=all(e.get("blocking", True) for e in cycle_edges),
            )
            norm = cp.normalized()
            if norm and norm not in seen_normalized:
                seen_normalized.add(norm)
                cycles.append(cp)
            return

        if node not in adj:
            return

        stack_set.add(node)
        stack_list.append(node)

        for neighbor, meta in adj.get(node, []):
            edge_stack.append(meta)
            self._dfs_cycles(
                neighbor, adj, stack_set, stack_list, edge_stack, cycles, seen_normalized
            )
            edge_stack.pop()

        stack_list.pop()
        stack_set.remove(node)

    # -- dangling -------------------------------------------------------------

    def _find_dangling(self, edges: list[CrossBoardEdge]) -> list[DanglingEdge]:
        dangling: list[DanglingEdge] = []
        boards_checked: dict[str, bool] = {}

        def _board_exists(board: str) -> bool:
            if board not in boards_checked:
                try:
                    boards_checked[board] = kb.kanban_db_path(board=board).exists()
                except Exception:
                    boards_checked[board] = False
            return boards_checked[board]

        for edge in edges:
            # Parent side
            if not _board_exists(edge.parent_board):
                dangling.append(
                    DanglingEdge(
                        edge=edge,
                        missing_side="parent",
                        reason=f"parent board '{edge.parent_board}' DB does not exist",
                    )
                )
                continue

            parent_status: str | None = None
            try:
                conn = kb.connect(board=edge.parent_board)
                try:
                    task = kb.get_task(conn, edge.parent_id)
                    if task is None:
                        dangling.append(
                            DanglingEdge(
                                edge=edge,
                                missing_side="parent",
                                reason=f"parent task '{edge.parent_id}' not found in board '{edge.parent_board}'",
                            )
                        )
                        continue
                    parent_status = task.status
                finally:
                    conn.close()
            except Exception as exc:
                dangling.append(
                    DanglingEdge(
                        edge=edge,
                        missing_side="parent",
                        reason=f"error reading parent board '{edge.parent_board}': {exc}",
                    )
                )
                continue

            # Child side
            if not _board_exists(edge.child_board):
                dangling.append(
                    DanglingEdge(
                        edge=edge,
                        missing_side="child",
                        reason=f"child board '{edge.child_board}' DB does not exist",
                    )
                )
                continue

            try:
                conn = kb.connect(board=edge.child_board)
                try:
                    task = kb.get_task(conn, edge.child_id)
                    if task is None:
                        dangling.append(
                            DanglingEdge(
                                edge=edge,
                                missing_side="child",
                                reason=f"child task '{edge.child_id}' not found in board '{edge.child_board}'",
                            )
                        )
                finally:
                    conn.close()
            except Exception as exc:
                dangling.append(
                    DanglingEdge(
                        edge=edge,
                        missing_side="child",
                        reason=f"error reading child board '{edge.child_board}': {exc}",
                    )
                )

        return dangling

    # -- contradictions -------------------------------------------------------

    def _find_contradictions(self, edges: list[CrossBoardEdge]) -> list[Contradiction]:
        contradictions: list[Contradiction] = []

        # Group edges by child so we can evaluate spawnability across all parents
        child_edges: dict[tuple[str, str], list[CrossBoardEdge]] = {}
        for edge in edges:
            child_edges.setdefault((edge.child_board, edge.child_id), []).append(edge)

        for (child_board, child_id), child_edges_list in child_edges.items():
            child_status = _task_status(child_board, child_id)
            if child_status is None:
                continue  # dangling handled separately

            # Separate blocking and non-blocking edges for this child
            blocking_edges = [e for e in child_edges_list if e.blocking]

            # Determine if child is spawnable (all blocking parents satisfied)
            all_blocking_satisfied = True
            unsatisfied_blocking: list[CrossBoardEdge] = []
            for edge in blocking_edges:
                parent_status = _task_status(edge.parent_board, edge.parent_id)
                if parent_status is None:
                    all_blocking_satisfied = False
                    continue
                required = edge.required_parent_statuses or DEFAULT_REQUIRED_PARENT_STATUSES
                if parent_status not in required:
                    all_blocking_satisfied = False
                    unsatisfied_blocking.append(edge)

            # Contradiction 1: child is ready/running but some blocking parent is unsatisfied
            if child_status in ("ready", "running") and unsatisfied_blocking:
                for edge in unsatisfied_blocking:
                    parent_status = _task_status(edge.parent_board, edge.parent_id)
                    required = edge.required_parent_statuses or DEFAULT_REQUIRED_PARENT_STATUSES
                    contradictions.append(
                        Contradiction(
                            edge=edge,
                            kind="unsatisfied_parent_child_ready_running",
                            parent_status=parent_status,
                            child_status=child_status,
                            reason=(
                                f"parent is '{parent_status}' (requires {required}) "
                                f"but child is '{child_status}'"
                            ),
                        )
                    )

            # Contradiction 2: all blocking parents satisfied but child is still stuck (todo/blocked)
            if all_blocking_satisfied and child_status not in ("ready", "running", "done", "archived"):
                # Only report if there is at least one blocking edge; otherwise the child may just have no deps
                if blocking_edges:
                    # Pick the first blocking edge for the report (all are satisfied)
                    edge = blocking_edges[0]
                    parent_status = _task_status(edge.parent_board, edge.parent_id)
                    required = edge.required_parent_statuses or DEFAULT_REQUIRED_PARENT_STATUSES
                    contradictions.append(
                        Contradiction(
                            edge=edge,
                            kind="satisfied_parent_child_stuck",
                            parent_status=parent_status,
                            child_status=child_status,
                            reason=(
                                f"parent is '{parent_status}' (satisfies {required}) "
                                f"but child is '{child_status}'"
                            ),
                        )
                    )

        return contradictions

    # -- provider failures ----------------------------------------------------

    def _find_provider_failures(self, edges: list[CrossBoardEdge]) -> list[ProviderFailure]:
        failures: list[ProviderFailure] = []
        for edge in edges:
            if not edge.blocking:
                continue
            try:
                blocker = self.provider._resolve_edge(edge)
                if blocker is None:
                    continue
                if not blocker.satisfied:
                    if "dangling or unreadable" in blocker.reason or "not found" in blocker.reason:
                        failures.append(
                            ProviderFailure(
                                edge=edge,
                                error_type="provider_dangling",
                                error_message=blocker.reason,
                            )
                        )
            except Exception as exc:
                failures.append(
                    ProviderFailure(
                        edge=edge,
                        error_type=exc.__class__.__name__,
                        error_message=str(exc),
                    )
                )
        return failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_status(board: str, task_id: str) -> str | None:
    try:
        conn = kb.connect(board=board)
        try:
            task = kb.get_task(conn, task_id)
            return task.status if task else None
        finally:
            conn.close()
    except Exception:
        return None


def _default_local_children_resolver(board: str, task_id: str) -> list[str]:
    try:
        conn = kb.connect(board=board)
        try:
            rows = conn.execute(
                "SELECT child_id FROM task_links WHERE parent_id = ?",
                (task_id,),
            ).fetchall()
            return [r["child_id"] for r in rows]
        finally:
            conn.close()
    except Exception:
        return []
