"""Shared raw-model proxy utilities for Hermes servers.

This module contains the pure functions and async handlers for the
OpenAI-compatible raw-model passthrough feature used by both:
  - gateway/platforms/api_server.py  (AIAgent-mode server)
  - hermes_cli/proxy/adapters/raw.py  (subscription proxy)

None of these symbols are coupled to the API server's auth/CORS/idempotency
machinery; keep it that way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_MODEL_PREFIX = "raw/"

RAW_PROVIDER_UNAVAILABLE_MESSAGE = "Raw model provider is unavailable."
RAW_PROVIDER_UNSUPPORTED_MESSAGE = "Raw model provider is unsupported or not configured."
RAW_PROVIDER_ROUTE_UNSUPPORTED_MESSAGE = "Raw model provider does not expose chat completions."
RAW_PROVIDER_ERROR_MESSAGE = "Raw model provider request failed."

_RAW_CHAT_COMPLETIONS_FORWARD_FIELDS: FrozenSet[str] = frozenset({
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "response_format",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "logit_bias",
    "user",
    "n",
    "stream",
})

_TRUE_REQUEST_BOOL_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_REQUEST_BOOL_STRINGS = frozenset({"0", "false", "no", "off"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def coerce_request_bool(value: Any, default: bool = False) -> bool:
    """Normalize boolean-like API payload values.

    External clients should send real JSON booleans, but some OpenAI-compatible
    frontends and middleware serialize flags like ``stream`` as strings.  Using
    Python truthiness on those values misroutes requests because ``"false"`` is
    still truthy.  Treat only explicit bool-ish scalars as booleans; everything
    else falls back to the caller's default.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_REQUEST_BOOL_STRINGS:
            return True
        if normalized in _FALSE_REQUEST_BOOL_STRINGS:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def raw_provider_exception_for_log(exc: BaseException) -> str:
    """Return provider exception text safe enough for server-side logs only.

    Raw proxy errors cross an auth boundary: callers authenticate to Hermes, but
    provider-resolution failures can contain credentials, auth file paths, and
    internal diagnostics.  Client responses therefore use stable generic error
    messages; logs may keep redacted details for operators.
    """
    try:
        text = str(exc)
    except Exception:
        text = f"{type(exc).__name__} (stringification failed)"
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except Exception:
        return f"{type(exc).__name__} (details unavailable)"


def openai_error(
    message: str,
    err_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI-style error envelope."""
    try:
        from agent.redact import redact_sensitive_text

        redacted_message = redact_sensitive_text(message, force=True)
    except Exception:
        redacted_message = message
    return {
        "error": {
            "message": redacted_message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


# ---------------------------------------------------------------------------
# Model-name parsing
# ---------------------------------------------------------------------------

def parse_raw_model_name(model_name: Any) -> Optional[Dict[str, str]]:
    """Return raw-provider routing info for ``raw/<provider>/<model>`` names.

    Non-string and non-raw model names return ``None`` so callers can keep the
    existing server-agent path.  Malformed raw names raise ``ValueError`` so a
    caller that explicitly asked for raw mode never silently falls back to the
    Hermes server-side AIAgent.
    """
    if not isinstance(model_name, str):
        return None
    raw_name = model_name.strip()
    if not raw_name.startswith(RAW_MODEL_PREFIX):
        return None
    suffix = raw_name[len(RAW_MODEL_PREFIX):]
    provider, sep, upstream_model = suffix.partition("/")
    provider = provider.strip().lower()
    upstream_model = upstream_model.strip()
    if not sep or not provider or not upstream_model:
        raise ValueError(
            "Raw model names must use raw/<provider>/<model>, "
            "for example raw/openai-codex/gpt-5.5."
        )
    return {
        "raw_model": raw_name,
        "provider": provider,
        "model": upstream_model,
    }


# ---------------------------------------------------------------------------
# Value normalisation
# ---------------------------------------------------------------------------

def jsonable_raw_value(value: Any) -> Any:
    """Convert SDK response objects/SimpleNamespaces into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable_raw_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_raw_value(v) for v in value]

    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return jsonable_raw_value(method(exclude_none=False))
            except TypeError:
                try:
                    return jsonable_raw_value(method())
                except Exception:
                    pass
            except Exception:
                pass

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return {
            str(k): jsonable_raw_value(v)
            for k, v in attrs.items()
            if not str(k).startswith("_")
        }
    return str(value)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def raw_chat_completion_payload(
    raw_response: Any,
    *,
    completion_id: str,
    created: int,
    model: str,
) -> Dict[str, Any]:
    """Normalize a provider ChatCompletion-like object to an OpenAI JSON dict."""
    payload = jsonable_raw_value(raw_response)
    if not isinstance(payload, dict):
        raise TypeError("Raw provider response was not a JSON object")
    payload.setdefault("id", completion_id)
    payload.setdefault("object", "chat.completion")
    payload.setdefault("created", created)
    payload.setdefault("model", model)
    choices = payload.get("choices")
    if isinstance(choices, list):
        for idx, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            choice.setdefault("index", idx)
            message = choice.get("message")
            if isinstance(message, dict):
                message.setdefault("role", "assistant")
    return payload


def raw_chat_completion_stream_chunk_from_final(raw_response: Any) -> Dict[str, Any]:
    """Convert a non-stream ChatCompletion response into one SSE chunk.

    Some provider adapters accept ``stream=True`` but still return a final
    ChatCompletion object (for example OAuth-backed compatibility adapters that
    synthesize the OpenAI surface).  Raw proxy clients such as DeepTutor consume
    the streaming Chat Completions shape, so translate the final ``message`` into
    a single ``delta`` chunk instead of failing the whole stream.
    """
    payload = jsonable_raw_value(raw_response)
    if not isinstance(payload, dict):
        raise TypeError("Raw provider response was not a JSON object")
    chunk: Dict[str, Any] = {
        "id": payload.get("id") or f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion.chunk",
        "created": payload.get("created") or int(time.time()),
        "model": payload.get("model") or "",
        "choices": [],
    }
    for choice_idx, choice in enumerate(payload.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        delta: Dict[str, Any] = {}
        role = message.get("role")
        if role:
            delta["role"] = role
        if message.get("content") is not None:
            delta["content"] = message.get("content")
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            normalized_tool_calls: List[Dict[str, Any]] = []
            for tool_idx, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                normalized_tool_call = dict(tool_call)
                normalized_tool_call.setdefault("index", tool_idx)
                normalized_tool_calls.append(normalized_tool_call)
            if normalized_tool_calls:
                delta["tool_calls"] = normalized_tool_calls
        chunk["choices"].append({
            "index": choice.get("index", choice_idx),
            "delta": delta,
            "finish_reason": choice.get("finish_reason"),
        })
    return chunk


def next_stream_item(iterator: Any) -> tuple[bool, Any]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


# ---------------------------------------------------------------------------
# Async handlers  (import aiohttp lazily where needed)
# ---------------------------------------------------------------------------

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False


async def handle_raw_chat_completions(
    request: "web.Request",
    body: Dict[str, Any],
    route: Dict[str, str],
    *,
    cors_headers: Optional[Dict[str, str]] = None,
) -> "web.Response":
    """Proxy ``raw/<provider>/<model>`` chat-completions to the provider.

    This path intentionally does not construct an ``AIAgent``.  Caller
    messages/tools remain caller-owned and returned tool calls are passed
    back for the client to execute.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for raw model proxy")

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return web.json_response(
            openai_error("Missing or invalid 'messages' field"),
            status=400,
        )

    provider = route["provider"]
    upstream_model = route["model"]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
    stream = coerce_request_bool(body.get("stream"), default=False)

    try:
        from agent.auxiliary_client import resolve_provider_client

        client, resolved_model = await asyncio.to_thread(
            resolve_provider_client,
            provider,
            model=upstream_model,
        )
    except Exception as exc:
        logger.warning(
            "Raw model proxy could not resolve provider %s: %s",
            provider,
            raw_provider_exception_for_log(exc),
        )
        return web.json_response(
            openai_error(
                RAW_PROVIDER_UNAVAILABLE_MESSAGE,
                err_type="server_error",
                code="raw_provider_unavailable",
                param="model",
            ),
            status=502,
        )

    if client is None or not resolved_model:
        return web.json_response(
            openai_error(
                RAW_PROVIDER_UNSUPPORTED_MESSAGE,
                code="raw_provider_unavailable",
                param="model",
            ),
            status=400,
        )

    try:
        create_fn = client.chat.completions.create
    except Exception:
        return web.json_response(
            openai_error(
                RAW_PROVIDER_ROUTE_UNSUPPORTED_MESSAGE,
                code="raw_route_unsupported",
                param="model",
            ),
            status=400,
        )

    upstream_kwargs: Dict[str, Any] = {
        key: body[key]
        for key in _RAW_CHAT_COMPLETIONS_FORWARD_FIELDS
        if key in body
    }
    upstream_kwargs["model"] = resolved_model
    upstream_kwargs["messages"] = messages
    upstream_kwargs["stream"] = stream

    idempotency_key = request.headers.get("Idempotency-Key")
    if idempotency_key:
        extra_headers = upstream_kwargs.get("extra_headers")
        if isinstance(extra_headers, dict):
            extra_headers = dict(extra_headers)
        else:
            extra_headers = {}
        extra_headers.setdefault("Idempotency-Key", idempotency_key)
        upstream_kwargs["extra_headers"] = extra_headers

    if stream:
        return await write_raw_sse_chat_completion(
            request,
            create_fn,
            upstream_kwargs,
            cors_headers=cors_headers,
        )

    try:
        raw_response = await asyncio.to_thread(create_fn, **upstream_kwargs)
        response_data = raw_chat_completion_payload(
            raw_response,
            completion_id=completion_id,
            created=created,
            model=resolved_model,
        )
    except Exception as exc:
        logger.warning(
            "Raw model proxy chat completion failed: %s",
            raw_provider_exception_for_log(exc),
        )
        return web.json_response(
            openai_error(
                RAW_PROVIDER_ERROR_MESSAGE,
                err_type="server_error",
                code="raw_provider_error",
            ),
            status=502,
        )

    return web.json_response(response_data)


async def write_raw_sse_chat_completion(
    request: "web.Request",
    create_fn: Any,
    upstream_kwargs: Dict[str, Any],
    *,
    cors_headers: Optional[Dict[str, str]] = None,
) -> "web.StreamResponse":
    """Forward provider ChatCompletion chunks as OpenAI SSE data lines."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for raw model proxy")

    sse_headers: Dict[str, str] = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if cors_headers:
        sse_headers.update(cors_headers)
    response = web.StreamResponse(status=200, headers=sse_headers)
    await response.prepare(request)

    try:
        stream_obj = await asyncio.to_thread(create_fn, **upstream_kwargs)
        if hasattr(stream_obj, "__aiter__"):
            async for chunk in stream_obj:
                data = json.dumps(jsonable_raw_value(chunk), ensure_ascii=False)
                await response.write(f"data: {data}\n\n".encode("utf-8"))
        else:
            try:
                iterator = iter(stream_obj)
            except TypeError:
                # Provider returned a final ChatCompletion despite stream=True
                # (OAuth compatibility adapters do this): emit one delta chunk.
                data = json.dumps(
                    raw_chat_completion_stream_chunk_from_final(stream_obj),
                    ensure_ascii=False,
                )
                await response.write(f"data: {data}\n\n".encode("utf-8"))
            else:
                while True:
                    has_item, chunk = await asyncio.to_thread(next_stream_item, iterator)
                    if not has_item:
                        break
                    data = json.dumps(jsonable_raw_value(chunk), ensure_ascii=False)
                    await response.write(f"data: {data}\n\n".encode("utf-8"))
        await response.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
        logger.info("Raw model proxy SSE client disconnected")
    except Exception as exc:
        logger.warning(
            "Raw model proxy SSE stream failed: %s",
            raw_provider_exception_for_log(exc),
        )
        try:
            error_chunk = openai_error(
                RAW_PROVIDER_ERROR_MESSAGE,
                err_type="server_error",
                code="raw_provider_error",
            )
            await response.write(
                f"event: error\ndata: {json.dumps(error_chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")
        except Exception:
            pass

    return response
