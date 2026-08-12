"""Safe session-title backfill/regeneration helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from agent.title_generator import generate_title

TitleGenerator = Callable[[str], str | None]


def _is_untitled(session: dict[str, Any]) -> bool:
    # Be conservative: without reliable provenance, a literal "Untitled" might
    # be a user-supplied manual title from before title_source existed.  Only
    # NULL/empty titles are safe backfill candidates.
    title = (session.get("title") or "").strip()
    return not title


def _is_auto_generated(session: dict[str, Any]) -> bool:
    return session.get("title_source") == "auto"


def _candidate_sessions(
    sessions: Iterable[dict[str, Any]],
    *,
    mode: str,
    ids: set[str] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for session in sessions:
        sid = str(session.get("id") or "")
        if ids is not None and sid not in ids:
            continue
        if mode == "untitled" and _is_untitled(session):
            candidates.append(session)
        elif mode == "auto-generated" and _is_auto_generated(session):
            candidates.append(session)
    return candidates


def _first_user_message(history) -> str:
    for msg in history or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _default_title_generator(context: str) -> str | None:
    return generate_title(context)


def retitle_sessions(
    db,
    *,
    mode: str = "untitled",
    apply: bool = False,
    source: str | None = None,
    limit: int = 50,
    ids: Iterable[str] | None = None,
    title_generator: TitleGenerator | None = None,
) -> dict[str, Any]:
    """Preview or apply regenerated titles for a narrow session set.

    ``apply=False`` is the default and has no DB side effects.  ``mode`` is
    intentionally conservative: untitled sessions are safe backfill candidates;
    existing titles are only regenerated when provenance says they were
    auto-generated.
    """
    if mode not in {"untitled", "auto-generated"}:
        raise ValueError("mode must be 'untitled' or 'auto-generated'")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    requested_ids = {str(x) for x in ids} if ids is not None else None
    generator = title_generator or _default_title_generator
    sessions = db.list_sessions_rich(source=source, limit=limit, include_archived=True)
    candidates = _candidate_sessions(sessions, mode=mode, ids=requested_ids)

    result_items: list[dict[str, Any]] = []
    title_source = "backfill" if mode == "untitled" else "auto"
    for session in candidates:
        session_id = str(session.get("id") or "")
        item = {
            "session_id": session_id,
            "old_title": session.get("title"),
            "new_title": None,
            "title_source": title_source,
            "applied": False,
            "error": None,
        }
        try:
            history = db.get_messages_as_conversation(session_id)
            context = _first_user_message(history)
            new_title = generator(context)
            item["new_title"] = new_title
            if apply and new_title:
                if mode == "untitled":
                    updated = db.set_backfill_session_title(session_id, new_title)
                else:
                    updated = db.set_auto_generated_session_title(session_id, new_title)
                if updated:
                    item["applied"] = True
                else:
                    item["error"] = "session no longer eligible for retitle"
        except Exception as exc:
            item["error"] = str(exc)
        result_items.append(item)

    return {
        "effect": "applied" if apply else "dry_run",
        "mode": mode,
        "source": source,
        "limit": limit,
        "candidates": result_items,
        "applied_count": sum(1 for item in result_items if item["applied"]),
    }
