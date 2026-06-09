"""kanban-cross-deps plugin — registry, provider, and CLI surfaces.

This file is intentionally minimal for the registry/storage slice.
The dependency provider and CLI surfaces will be added in downstream cards.
"""

from __future__ import annotations

from plugins.kanban_cross_deps.models import CrossBoardEdge, VALID_EDGE_KINDS
from plugins.kanban_cross_deps.store import CrossBoardRegistry

__all__ = [
    "CrossBoardEdge",
    "CrossBoardRegistry",
    "VALID_EDGE_KINDS",
]
