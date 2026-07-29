"""Auto-generate short, human-browsable session titles.

Runs asynchronously after responses are delivered so title generation never adds
latency to the user-facing reply.  The generator uses a compact structured
context window rather than only the first exchange, so titles stay specific when
the real task emerges after tool use or a follow-up.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

# Validation callback: () -> bool. Called right before the LLM request in
# generate_title(). Return False to skip — e.g. the user switched models
# after this background thread captured its runtime snapshot, and sending
# the request would reload a model the runtime already evicted (#19027).
RuntimeValidator = Callable[[], bool]

_TITLE_PROMPT = (
    "Generate a concise, human-browsable title for this Hermes session. "
    "Prefer the specific object, repo, bug, system, artifact, or decision over "
    "generic task words. Write the title in the same language the user is "
    "writing in. Include enough distinguishing nouns to tell this session apart "
    "from adjacent sessions. Avoid vague titles like 'Troubleshooting Issue', "
    "'Code Review', 'Hermes Debugging', or 'Session Title'. Return ONLY 4-10 "
    "words: no quotes, no prefix, no trailing punctuation."
)

_TITLE_PROMPT_PINNED_LANGUAGE = (
    "Generate a concise, human-browsable title for this Hermes session. "
    "Prefer the specific object, repo, bug, system, artifact, or decision over "
    "generic task words. Write the title in {language}. Include enough "
    "distinguishing nouns to tell this session apart from adjacent sessions. "
    "Avoid vague titles like 'Troubleshooting Issue', 'Code Review', "
    "'Hermes Debugging', or 'Session Title'. Return ONLY 4-10 words: no quotes, "
    "no prefix, no trailing punctuation."
)

_MAX_FIELD_CHARS = 900
_MAX_CONTEXT_CHARS = 4000


def _title_language() -> str:
    """Return configured title language, or empty string to match the user."""
    try:
        from hermes_cli.config import load_config_readonly

        return str(
            ((load_config_readonly() or {}).get("auxiliary") or {})
            .get("title_generation", {})
            .get("language", "")
        ).strip()
    except Exception:
        return ""


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("text"):
                    parts.append(str(item.get("text")))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(content)


def _clean_snippet(value: Any, max_chars: int = _MAX_FIELD_CHARS) -> str:
    text = re.sub(r"\s+", " ", _stringify_content(value)).strip()
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def build_title_context(
    conversation_history: list[dict[str, Any]] | None,
    *,
    current_user_message: Any = None,
    current_assistant_response: Any = None,
    session: dict[str, Any] | None = None,
) -> str:
    """Build a compact, structured title context for the auxiliary LLM.

    The title generator needs enough information to distinguish sessions, but
    not a full transcript.  This context favors stable human-browsing signals:
    source/workspace/model, first user request, latest user request, latest
    assistant answer, and tool names used during the session.
    """
    history = [m for m in (conversation_history or []) if isinstance(m, dict)]
    user_messages = [m for m in history if m.get("role") == "user"]
    assistant_messages = [m for m in history if m.get("role") == "assistant"]

    first_user = user_messages[0].get("content") if user_messages else current_user_message
    latest_user = current_user_message
    if latest_user is None and user_messages:
        latest_user = user_messages[-1].get("content")

    # A /skill turn is expanded into a large scaffolding message. Preserve the
    # rich context path while making its user-facing request match the compact
    # legacy title path, so skill sessions are titled after the request rather
    # than after the embedded SKILL.md body.
    first_user = _summarize_user_message(_stringify_content(first_user))
    latest_user = _summarize_user_message(_stringify_content(latest_user))

    recent_assistant = current_assistant_response
    if recent_assistant is None and assistant_messages:
        recent_assistant = assistant_messages[-1].get("content")

    tool_names: list[str] = []
    seen_tools: set[str] = set()
    for message in history:
        name = message.get("tool_name")
        if name and name not in seen_tools:
            seen_tools.add(name)
            tool_names.append(str(name))
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function") if isinstance(tool_call, dict) else None
            call_name = fn.get("name") if isinstance(fn, dict) else None
            if call_name and call_name not in seen_tools:
                seen_tools.add(call_name)
                tool_names.append(str(call_name))

    lines: list[str] = []
    if session:
        source = session.get("source")
        cwd = session.get("cwd")
        model = session.get("model")
        if source:
            lines.append(f"Source: {_clean_snippet(source, 120)}")
        if cwd:
            lines.append(f"Workspace: {_clean_snippet(cwd, 240)}")
        if model:
            lines.append(f"Model: {_clean_snippet(model, 160)}")

    if first_user:
        lines.append(f"First user request: {_clean_snippet(first_user)}")
    if latest_user and _clean_snippet(latest_user) != _clean_snippet(first_user):
        lines.append(f"Latest user request: {_clean_snippet(latest_user)}")
    elif latest_user:
        lines.append(f"Latest user request: {_clean_snippet(latest_user)}")
    if recent_assistant:
        lines.append(f"Recent assistant answer: {_clean_snippet(recent_assistant)}")
    if tool_names:
        lines.append(f"Tools used: {', '.join(tool_names[:12])}")

    context = "\n".join(lines).strip()
    if len(context) > _MAX_CONTEXT_CHARS:
        return context[: _MAX_CONTEXT_CHARS - 1].rstrip() + "…"
    return context


def _auto_title_enabled() -> bool:
    """Return whether automatic session title generation is enabled."""
    try:
        # Lazy imports, matching _title_language(): title_generator is imported
        # from agent code paths where a module-level hermes_cli import risks
        # circularity, and the read-only loader avoids config-migration writes.
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        config = load_config_readonly()
        title_config = (config.get("auxiliary") or {}).get("title_generation") or {}
        return is_truthy_value(title_config.get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


def _summarize_user_message(user_message: str) -> str:
    """Collapse a slash-skill-expanded turn back to what the user typed.

    A ``/skill`` invocation expands into a message that embeds the whole skill
    body, so feeding it to the titler verbatim titles the session after the
    *skill's* prose — "Kick off a task in a fresh isolated git worktree" — not
    after the user's request. Reuse the canonical scaffolding parser so the
    model sees ``/work — fix the title leak`` instead.
    """
    if not user_message:
        return ""
    try:
        from agent.skill_commands import describe_skill_invocation

        described = describe_skill_invocation(user_message)
    except Exception:
        logger.debug("Skill-scaffolding summary failed; titling raw", exc_info=True)
        return user_message
    return described if described is not None else user_message


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_context: str | None = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Generate a session title.

    ``title_context`` is preferred when supplied.  The legacy
    ``user_message``/``assistant_response`` path remains for older callers and
    tests, but only includes short first-exchange snippets.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    auxiliary call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.

    ``runtime_validator`` is called right before the LLM request. If it
    returns False (e.g. the user's model was switched since the background
    thread captured its runtime snapshot), the call is skipped silently —
    no request is sent, so a stale title request can't reload a model the
    runtime already unloaded (#19027).
    """
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None

    if runtime_validator is not None:
        try:
            if not runtime_validator():
                logger.debug("Title generation skipped: runtime validator returned False")
                return None
        except Exception:
            # Fail open: a broken validator must not disable titling.
            logger.debug("Title runtime validator raised; proceeding", exc_info=True)

    if title_context:
        prompt_body = title_context[:_MAX_CONTEXT_CHARS]
    else:
        # Truncate long messages to keep the request small on the legacy path.
        user_snippet = _summarize_user_message(user_message)[:500]
        assistant_snippet = assistant_response[:500] if assistant_response else ""
        prompt_body = f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"

    language = _title_language()
    prompt = _TITLE_PROMPT_PINNED_LANGUAGE.format(language=language) if language else _TITLE_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": prompt_body},
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=timeout,
            main_runtime=main_runtime,
        )
        content = response.choices[0].message.content or ""
        # Strip thinking/reasoning blocks that think-enabled models
        # (MiniMax M2.7, DeepSeek, etc.) emit even for simple prompts like
        # title generation. Without this the raw <think>...</think> XML
        # leaks into session titles. Reuses the canonical scrubber so all
        # tag variants (unterminated blocks, orphan closes, mixed case)
        # are handled, not just a single literal <think> pair.
        from agent.agent_runtime_helpers import strip_think_blocks
        title = strip_think_blocks(None, content).strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # A title is one line. A model that ignores "return ONLY the title" and
        # answers the prompt instead (a shell transcript, a bulleted plan) would
        # otherwise be stored verbatim and truncated mid-command. Keep the first
        # non-empty line — the closest thing to a title in that response.
        title = next((line.strip() for line in title.splitlines() if line.strip()), "")
        title = title.rstrip(".。!！?？")
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        # Log at WARNING so this shows up in agent.log without debug mode.
        # Full detail at debug level for operators who need the stack.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)
        return None


def _session_allows_auto_retitle(session_db, session_id: str) -> bool:
    try:
        session = session_db.get_session(session_id)
    except Exception:
        return False
    if not isinstance(session, dict):
        return False
    if session.get("title_source") != "auto":
        return False
    try:
        revisions = int(session.get("title_revision_count") or 0)
    except (TypeError, ValueError):
        revisions = 0
    return revisions <= 1


def _persist_session_title(session_db, session_id, title):
    """Persist a generated title, recovering from duplicate-title collisions.

    The write goes through ``set_auto_title_if_empty`` (predicate + write in
    one transaction) so a manual ``/title`` set while LLM generation was in
    flight is never overwritten — a plain ``set_session_title`` fallback keeps
    older stores working. ``set_session_title`` raises ValueError when the
    title would collide with another session (the unique-title index). Rather
    than swallow it and leave the session untitled (#50537), append a #N
    suffix via get_next_title_in_lineage() when the store supports lineage
    dedup; otherwise re-raise so the caller can decide.

    Returns the title actually persisted, or None when a concurrent manual
    title won the race (nothing was written).
    """
    atomic_fn = getattr(session_db, "set_auto_title_if_empty", None)

    def _set(t):
        if atomic_fn is not None:
            if not atomic_fn(session_id, t):
                # Predicate failed: a title appeared while generation was in
                # flight (manual /title wins), or the session vanished.
                logger.debug(
                    "Skipping auto-generated session title because a title "
                    "was set while generation was in flight"
                )
                return None
            return t
        ok = session_db.set_session_title(session_id, t)
        if ok is False:
            raise RuntimeError(
                f"session {session_id} not found when storing title"
            )
        return t

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    title_context: str | None = None,
    allow_retitle: bool = False,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and set a session title if safe.

    Called in a background thread after an exchange completes.  Existing manual
    titles are never overwritten.  Existing auto-generated titles may be
    improved once when ``allow_retitle`` is true.

    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    - runtime_validator returns False (model was switched)

    Never lets an exception escape: this is a daemon-thread target, and an
    escaping exception would spray a raw traceback into the user's terminal
    via the default threading excepthook. The canonical trigger is the
    post-``hermes update`` stale-module window, where this function's lazy
    imports read NEW source from disk while already-cached modules
    (``agent.portal_tags`` etc.) are still the OLD version — the resulting
    ImportError repeats on every auto-title attempt until the long-running
    process restarts.
    """
    try:
        _auto_title_session(
            session_db,
            session_id,
            user_message,
            assistant_response,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
            title_callback=title_callback,
            title_context=title_context,
            allow_retitle=allow_retitle,
            runtime_validator=runtime_validator,
        )
    except Exception as e:
        # WARNING (not debug) so operators see it in agent.log; the message
        # names the likely cause so "restart the process" is discoverable.
        logger.warning(
            "Auto-title failed (harmless; if this started after an update, "
            "restart the running Hermes process): %s",
            e,
        )
        logger.debug("Auto-title traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Auto-title failure_callback raised", exc_info=True)


def _auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    title_context: str | None = None,
    allow_retitle: bool = False,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Body of :func:`auto_title_session` — see its docstring."""
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing and not (allow_retitle and _session_allows_auto_retitle(session_db, session_id)):
            return
    except Exception:
        return

    # This runs on a bare daemon thread spawned AFTER the turn's ambient
    # conversation context was reset, so publish it here from the session id
    # we already hold — the title-generation LLM call then carries the same
    # ``conversation=`` Portal tag as the turn it titles. Root-of-lineage for
    # consistency with the agent loop (a no-op on first exchange, where
    # titling happens, but correct if this ever runs on a continuation).
    from agent.aux_accounting import set_accounting_context
    from agent.portal_tags import set_conversation_context

    conversation_id = session_id
    try:
        conversation_id = session_db.get_conversation_root(session_id) or session_id
    except Exception:
        pass
    set_conversation_context(conversation_id)
    # Same for the accounting context, so the title call's token usage is
    # recorded against this session (task='title_generation', #23270).
    set_accounting_context(session_db, session_id)

    title = generate_title(
        user_message,
        assistant_response,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
        title_context=title_context,
        runtime_validator=runtime_validator,
    )
    if not title:
        return

    try:
        persisted = _persist_session_title(session_db, session_id, title)
        if persisted is None:
            return
        logger.debug("Auto-generated session title: %s", persisted)
        if title_callback is not None:
            try:
                title_callback(persisted)
            except Exception:
                logger.debug("Auto-title callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Fire-and-forget title generation after early exchanges.

    Generates a provisional title on the first two user exchanges, then allows
    one improvement on the third exchange only when the existing title is known
    to be auto-generated.
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    history = conversation_history or []
    user_msg_count = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user")
    allow_retitle = False
    if user_msg_count <= 2:
        allow_retitle = False
    elif user_msg_count == 3 and _session_allows_auto_retitle(session_db, session_id):
        allow_retitle = True
    else:
        return

    session_meta = None
    try:
        candidate = session_db.get_session(session_id)
        if isinstance(candidate, dict):
            session_meta = candidate
    except Exception:
        session_meta = None

    title_context = build_title_context(
        history,
        current_user_message=user_message,
        current_assistant_response=assistant_response,
        session=session_meta,
    )
    # Config read comes after the cheap first-exchange guard so the file
    # isn't touched on every subsequent turn of a long session.
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        kwargs={
            "title_context": title_context,
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
            "allow_retitle": allow_retitle,
            "runtime_validator": runtime_validator,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
