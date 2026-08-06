"""Kanban budget telemetry: per-run overhead envelope and normalized complexity proxy.

Purely additive diagnostics used by the Kanban engine to separate per-turn
harness overhead from work attributable to a task. The telemetry is stored in
``task_events`` payloads and ``task_runs.metadata`` so downstream tooling can
plot ``iterations / complexity`` over time and detect regressions in per-turn
overhead independently of task difficulty.

This module has no runtime behavior change: it only produces dictionaries that
callers may include in events and run metadata.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def _count_code_blocks(text: str) -> int:
    """Count fenced code blocks and inline backtick runs."""
    if not text:
        return 0
    fenced = len(re.findall(r"```[a-zA-Z0-9]*\n", text))
    inline = len(re.findall(r"`[^`]+`", text))
    return fenced + inline


def _count_links(text: str) -> int:
    """Count Markdown links, bare http(s) URLs, and task/comment refs."""
    if not text:
        return 0
    md_links = len(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text))
    bare_urls = len(re.findall(r"https?://\S+", text))
    card_refs = len(re.findall(r"\bt_[a-f0-9]{8,}\b", text))
    return md_links + bare_urls + card_refs


def estimate_task_complexity(
    title: Optional[str] = None,
    body: Optional[str] = None,
    parents: Optional[list] = None,
    children: Optional[list] = None,
) -> Dict[str, Any]:
    """Return a normalized complexity proxy for a Kanban task.

    The score is deliberately coarse and stable: it treats the task as a
    document-plus-graph problem. Inputs are bounded so extreme cards cannot
    blow up the ratio used for regression detection.

    Components:
      * words and lines in the body (long specs cost more)
      * fenced code blocks / inline backticks (implementation tasks carry code)
      * links and cross-card references (integration surface)
      * parent and child links (coordination overhead)

    The score is normalized so a trivial one-line card is ~1.0 and a large
    multi-parent design card may be ~20-40. The exact formula is versioned in
    the returned dict so historical ratios remain interpretable.
    """
    title = title or ""
    body = body or ""
    parents = parents or []
    children = children or []

    words = len(body.split())
    lines = len(body.splitlines())
    code_blocks = _count_code_blocks(body)
    links = _count_links(body)

    # Title contributes a small constant; very short titles are common.
    title_words = len(title.split())

    # Bound each term so a pathological card cannot dominate the metric.
    score = (
        1.0
        + min(title_words / 20.0, 5.0)
        + min(words / 200.0, 20.0)
        + min(lines / 100.0, 15.0)
        + min(code_blocks * 1.5, 20.0)
        + min(links * 0.75, 10.0)
        + min(len(parents) * 2.0, 10.0)
        + min(len(children) * 1.5, 10.0)
    )

    return {
        "score": round(max(1.0, score), 2),
        "version": 1,
        "words": words,
        "lines": lines,
        "code_blocks": code_blocks,
        "links": links,
        "title_words": title_words,
        "parents": len(parents),
        "children": len(children),
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _count_tool_turns(messages: list) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, dict)
        and m.get("role") == "assistant"
        and m.get("tool_calls")
    )


def _count_assistant_messages(messages: list) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, dict) and m.get("role") == "assistant"
    )


def _count_user_messages(messages: list) -> int:
    return sum(
        1
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    )


def _last_turn_usage(agent: Any) -> Dict[str, int]:
    """Extract the most recent turn's usage dict when the host recorded it."""
    usage = getattr(agent, "_last_turn_usage", None)
    if not isinstance(usage, dict):
        return {}
    keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    }
    return {k: _safe_int(usage.get(k)) for k in keys if usage.get(k) is not None}


def build_overhead_envelope(
    agent: Any,
    api_call_count: int,
    messages: Optional[list] = None,
    turn_exit_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a serializable overhead envelope for one task run.

    The envelope captures counts that let operators distinguish "the task was
    hard" from "the harness spent too many turns on overhead". All token counts
    are best-effort: the agent records what it observed; missing values default
    to 0 rather than raising.
    """
    messages = messages or []
    budget = getattr(agent, "iteration_budget", None)
    compressor = getattr(agent, "context_compressor", None)

    envelope: Dict[str, Any] = {
        "api_call_count": _safe_int(api_call_count),
        "max_iterations": _safe_int(getattr(agent, "max_iterations", None)),
        "budget_used": _safe_int(getattr(budget, "used", None) if budget else None),
        "budget_max": _safe_int(getattr(budget, "max_total", None) if budget else None),
        "tool_call_turns": _count_tool_turns(messages),
        "assistant_messages": _count_assistant_messages(messages),
        "user_messages": _count_user_messages(messages),
        "message_count": len(messages),
        "input_tokens": _safe_int(getattr(agent, "session_input_tokens", None)),
        "output_tokens": _safe_int(getattr(agent, "session_output_tokens", None)),
        "total_tokens": _safe_int(getattr(agent, "session_total_tokens", None)),
        "cache_read_tokens": _safe_int(getattr(agent, "session_cache_read_tokens", None)),
        "cache_write_tokens": _safe_int(getattr(agent, "session_cache_write_tokens", None)),
        "reasoning_tokens": _safe_int(getattr(agent, "session_reasoning_tokens", None)),
        "last_prompt_tokens": _safe_int(
            getattr(compressor, "last_prompt_tokens", None) if compressor else None
        ),
        "last_turn_usage": _last_turn_usage(agent),
        "exit_reason": turn_exit_reason,
    }

    # Defensive: drop values that are still 0 when the underlying source is
    # genuinely absent, but keep explicit zeros where the source existed.
    # This keeps the payload compact without losing meaningful negatives.
    for key in list(envelope):
        if envelope[key] is None:
            envelope[key] = 0

    return envelope


def _is_serializable(value: Any) -> bool:
    """Check whether a value round-trips through JSON."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def merge_telemetry_into_payload(
    payload: Dict[str, Any],
    *,
    complexity_proxy: Optional[Dict[str, Any]] = None,
    overhead_envelope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge validated telemetry into an event payload in-place.

    Values that cannot be JSON-serialized are skipped so a malformed agent
    attribute cannot break the Kanban event log.
    """
    if complexity_proxy is not None and _is_serializable(complexity_proxy):
        payload["complexity_proxy"] = complexity_proxy
    if overhead_envelope is not None and _is_serializable(overhead_envelope):
        payload["overhead_envelope"] = overhead_envelope
    return payload
