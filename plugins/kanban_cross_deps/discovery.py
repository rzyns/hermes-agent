"""Read-only discovery of cross-board dependency candidates.

Scans task bodies, comments, results, and artifact paths for board-qualified
task references and classifies candidates without ever mutating the canonical
registry. Promotion to canonical edges is an explicit separate operation.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

from hermes_cli import kanban_db as kb

from plugins.kanban_cross_deps.models import VALID_EDGE_KINDS
from plugins.kanban_cross_deps.store import CrossBoardRegistry

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Board-qualified reference: board_slug/t_xxxxxxxx
_BOARD_QUALIFIED_RE = re.compile(
    r"(?P<board>[a-zA-Z0-9_-]+)/(?P<task_id>t_[a-f0-9]{8})",
    re.IGNORECASE,
)

# Bare task id: t_xxxxxxxx (may be same-board or cross-board)
_BARE_TASK_ID_RE = re.compile(
    r"(?<!/)(?P<task_id>t_[a-f0-9]{8})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Candidate dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DependencyCandidate:
    """An inferred cross-board dependency candidate, never canonical by itself."""

    child_board: str
    child_id: str
    referenced_board: str
    referenced_id: str
    source_location: str  # e.g. "body", "result", "comment:3", "artifact_path"
    context_snippet: str  # surrounding text (~120 chars)
    inferred_kind: str
    confidence: float  # 0.0–1.0
    status: str  # "inferred", "already_canonical", "dangling", "ambiguous"
    canonical_edge_id: str | None = None  # set when already_canonical

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_board": self.child_board,
            "child_id": self.child_id,
            "referenced_board": self.referenced_board,
            "referenced_id": self.referenced_id,
            "source_location": self.source_location,
            "context_snippet": self.context_snippet,
            "inferred_kind": self.inferred_kind,
            "confidence": self.confidence,
            "status": self.status,
            "canonical_edge_id": self.canonical_edge_id,
        }


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------

class CandidateDiscovery:
    """Read-only scanner for board-qualified task references."""

    def __init__(
        self,
        registry: CrossBoardRegistry | None = None,
    ) -> None:
        self.registry = registry or CrossBoardRegistry()

    # -- public API ----------------------------------------------------------

    def discover(
        self,
        child_board: str,
        child_id: str | None = None,
        max_snippet_len: int = 120,
    ) -> list[DependencyCandidate]:
        """Scan tasks in *child_board* for cross-board references.

        If *child_id* is given, scan only that task.  Returns a list of
        candidates sorted by confidence descending.
        """
        candidates: list[DependencyCandidate] = []

        conn = kb.connect(board=child_board)
        try:
            if child_id is not None:
                task = kb.get_task(conn, child_id)
                tasks = [task] if task else []
            else:
                cur = conn.execute(
                    "SELECT id, title, body, result FROM tasks ORDER BY created_at DESC"
                )
                tasks = [dict(r) for r in cur.fetchall()]

            for task in tasks:
                if hasattr(task, "id") and not isinstance(task, dict):
                    task = dataclasses.asdict(task)
                tid = str(task["id"])
                candidates.extend(
                    self._scan_task(
                        conn=conn,
                        child_board=child_board,
                        child_id=tid,
                        task=task,
                        max_snippet_len=max_snippet_len,
                    )
                )
        finally:
            conn.close()

        # Deduplicate by (child_board, child_id, referenced_board, referenced_id, source_location)
        seen: set[tuple[str, str, str, str, str]] = set()
        deduped: list[DependencyCandidate] = []
        for c in sorted(candidates, key=lambda x: x.confidence, reverse=True):
            key = (c.child_board, c.child_id, c.referenced_board, c.referenced_id, c.source_location)
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped

    # -- internal scanning ---------------------------------------------------

    def _scan_task(
        self,
        conn,
        child_board: str,
        child_id: str,
        task,
        max_snippet_len: int,
    ) -> list[DependencyCandidate]:
        """Scan a single task row for cross-board references."""
        # Normalize Task dataclass to dict for uniform access
        if hasattr(task, "id") and not isinstance(task, dict):
            task = dataclasses.asdict(task)
        candidates: list[DependencyCandidate] = []

        # Body
        body = task.get("body") or ""
        candidates.extend(
            self._scan_text(
                text=body,
                child_board=child_board,
                child_id=child_id,
                source_location="body",
                max_snippet_len=max_snippet_len,
            )
        )

        # Result
        result = task.get("result") or ""
        candidates.extend(
            self._scan_text(
                text=result,
                child_board=child_board,
                child_id=child_id,
                source_location="result",
                max_snippet_len=max_snippet_len,
            )
        )

        # Title
        title = task.get("title") or ""
        candidates.extend(
            self._scan_text(
                text=title,
                child_board=child_board,
                child_id=child_id,
                source_location="title",
                max_snippet_len=max_snippet_len,
            )
        )

        # Comments
        comment_rows = conn.execute(
            "SELECT id, body FROM task_comments WHERE task_id = ? ORDER BY created_at",
            (child_id,),
        ).fetchall()
        for row in comment_rows:
            candidates.extend(
                self._scan_text(
                    text=row["body"] or "",
                    child_board=child_board,
                    child_id=child_id,
                    source_location=f"comment:{row['id']}",
                    max_snippet_len=max_snippet_len,
                )
            )

        return candidates

    def _scan_text(
        self,
        text: str,
        child_board: str,
        child_id: str,
        source_location: str,
        max_snippet_len: int,
    ) -> list[DependencyCandidate]:
        candidates: list[DependencyCandidate] = []
        if not text:
            return candidates

        # 1. Board-qualified references — highest confidence
        for m in _BOARD_QUALIFIED_RE.finditer(text):
            ref_board = m.group("board").lower()
            ref_id = m.group("task_id").lower()
            # Skip self-references
            if ref_board == child_board and ref_id == child_id:
                continue
            snippet = _extract_snippet(text, m.start(), m.end(), max_snippet_len)
            kind = _infer_kind(snippet)
            status, canonical_id, confidence = self._classify(
                child_board=child_board,
                child_id=child_id,
                ref_board=ref_board,
                ref_id=ref_id,
                base_confidence=0.9,
            )
            candidates.append(
                DependencyCandidate(
                    child_board=child_board,
                    child_id=child_id,
                    referenced_board=ref_board,
                    referenced_id=ref_id,
                    source_location=source_location,
                    context_snippet=snippet,
                    inferred_kind=kind,
                    confidence=confidence,
                    status=status,
                    canonical_edge_id=canonical_id,
                )
            )

        # 2. Bare task IDs — only consider if they exist on a *different* board
        for m in _BARE_TASK_ID_RE.finditer(text):
            ref_id = m.group("task_id").lower()
            # Skip self-reference
            if ref_id == child_id:
                continue
            # Check if this bare ID exists on a different board
            boards_found = _boards_for_task_id(ref_id, exclude_board=child_board)
            if not boards_found:
                continue
            if len(boards_found) > 1:
                # Ambiguous: exists on multiple boards
                snippet = _extract_snippet(text, m.start(), m.end(), max_snippet_len)
                kind = _infer_kind(snippet)
                for ref_board in boards_found:
                    status, canonical_id, confidence = self._classify(
                        child_board=child_board,
                        child_id=child_id,
                        ref_board=ref_board,
                        ref_id=ref_id,
                        base_confidence=0.4,
                    )
                    candidates.append(
                        DependencyCandidate(
                            child_board=child_board,
                            child_id=child_id,
                            referenced_board=ref_board,
                            referenced_id=ref_id,
                            source_location=source_location,
                            context_snippet=snippet,
                            inferred_kind=kind,
                            confidence=confidence,
                            status="ambiguous",
                            canonical_edge_id=canonical_id,
                        )
                    )
            else:
                ref_board = boards_found[0]
                snippet = _extract_snippet(text, m.start(), m.end(), max_snippet_len)
                kind = _infer_kind(snippet)
                status, canonical_id, confidence = self._classify(
                    child_board=child_board,
                    child_id=child_id,
                    ref_board=ref_board,
                    ref_id=ref_id,
                    base_confidence=0.6,
                )
                candidates.append(
                    DependencyCandidate(
                        child_board=child_board,
                        child_id=child_id,
                        referenced_board=ref_board,
                        referenced_id=ref_id,
                        source_location=source_location,
                        context_snippet=snippet,
                        inferred_kind=kind,
                        confidence=confidence,
                        status=status,
                        canonical_edge_id=canonical_id,
                    )
                )

        return candidates

    def _classify(
        self,
        child_board: str,
        child_id: str,
        ref_board: str,
        ref_id: str,
        base_confidence: float,
    ) -> tuple[str, str | None, float]:
        """Return (status, canonical_edge_id_or_None, adjusted_confidence)."""
        # Check canonical registry
        canonical = self.registry.list_edges(
            child_board=child_board,
            child_id=child_id,
            parent_board=ref_board,
            parent_id=ref_id,
            limit=1,
        )
        if canonical:
            return "already_canonical", canonical[0].id, base_confidence

        # Check dangling
        board_exists = _board_db_exists(ref_board)
        if not board_exists:
            return "dangling", None, base_confidence * 0.5

        task_exists = _task_exists(ref_board, ref_id)
        if not task_exists:
            return "dangling", None, base_confidence * 0.5

        return "inferred", None, base_confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_snippet(text: str, start: int, end: int, max_len: int) -> str:
    """Extract surrounding text with the match centred."""
    half = max_len // 2
    snippet_start = max(0, start - half)
    snippet_end = min(len(text), end + half)
    snippet = text[snippet_start:snippet_end]
    if snippet_start > 0:
        snippet = "..." + snippet
    if snippet_end < len(text):
        snippet = snippet + "..."
    # Collapse whitespace for readability
    return " ".join(snippet.split())


def _infer_kind(snippet: str) -> str:
    """Heuristic kind inference from context snippet."""
    lowered = snippet.lower()
    if any(w in lowered for w in ("block", "blocks", "blocking", "prevent")):
        return "blocks"
    if any(w in lowered for w in ("depend", "needs", "requires", "wait")):
        return "depends_on"
    if any(w in lowered for w in ("decision", "approve", "gate", "sign-off")):
        return "depends_on_decision"
    if any(w in lowered for w in ("inform", "see", "refer", "check")):
        return "informed_by"
    if any(w in lowered for w in ("research", "investigate", "study")):
        return "derived_from_research"
    if any(w in lowered for w in ("feed", "output", "result")):
        return "feeds"
    if any(w in lowered for w in ("supersed", "replace", "obsolete")):
        return "supersedes"
    return "related"


def _boards_for_task_id(task_id: str, exclude_board: str | None = None) -> list[str]:
    """Find which boards contain *task_id*.  Expensive — scans DB directory."""
    found: list[str] = []
    try:
        base = kb.boards_root()
        if not base.exists():
            return found
        for board_dir in base.iterdir():
            if not board_dir.is_dir():
                continue
            board = board_dir.name
            if exclude_board and board == exclude_board:
                continue
            db_file = board_dir / "kanban.db"
            if not db_file.exists():
                continue
            try:
                conn = kb.connect(board=board)
                try:
                    row = conn.execute(
                        "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
                    ).fetchone()
                    if row:
                        found.append(board)
                finally:
                    conn.close()
            except Exception:
                pass
    except Exception:
        pass
    return found


def _board_db_exists(board: str) -> bool:
    try:
        return kb.kanban_db_path(board=board).exists()
    except Exception:
        return False


def _task_exists(board: str, task_id: str) -> bool:
    try:
        conn = kb.connect(board=board)
        try:
            row = conn.execute(
                "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False
