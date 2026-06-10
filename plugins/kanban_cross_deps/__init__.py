"""kanban-cross-deps plugin — registry, provider, and CLI surfaces.

This file is intentionally minimal for the registry/storage slice.
The dependency provider and CLI surfaces will be added in downstream cards.
"""

from __future__ import annotations

from plugins.kanban_cross_deps.cli import kanban_cross_deps_command, register_cli
from plugins.kanban_cross_deps.diagnostics import CrossBoardDiagnostics
from plugins.kanban_cross_deps.discovery import CandidateDiscovery, DependencyCandidate
from plugins.kanban_cross_deps.models import CrossBoardEdge, VALID_EDGE_KINDS
from plugins.kanban_cross_deps.provider import CrossBoardDependencyProvider
from plugins.kanban_cross_deps.store import CrossBoardRegistry

__all__ = [
    "CrossBoardEdge",
    "CrossBoardRegistry",
    "CrossBoardDependencyProvider",
    "CrossBoardDiagnostics",
    "CandidateDiscovery",
    "DependencyCandidate",
    "VALID_EDGE_KINDS",
    "kanban_cross_deps_command",
    "register_cli",
    "register",
]


def register(ctx) -> None:
    """Register CLI command and Kanban dependency provider for the plugin.

    Called once by the plugin loader when the plugin is enabled via
    ``plugins.enabled`` in config.yaml.
    """
    ctx.register_cli_command(
        name="kanban-cross-deps",
        help="Cross-board Kanban dependency registry",
        setup_fn=register_cli,
        handler_fn=kanban_cross_deps_command,
        description=(
            "Manage canonical cross-board dependency edges: add, remove, list, "
            "and inspect blocking status."
        ),
    )

    # Bridge the registry to the core seam so cross-board blocking edges
    # participate in scheduler invariants (claim + recompute_ready).
    provider = CrossBoardDependencyProvider()
    ctx.register_kanban_dependency_provider(provider)
