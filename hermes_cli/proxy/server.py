"""HTTP server that forwards OpenAI-compatible requests to a configured upstream.

A credential-attaching forwarder: request/response bodies are never mediated, logged, or rewritten.
The one shim: after a *clean* upstream EOF, a ``text/event-stream`` response that carried a terminal
``finish_reason`` or ``lastOne: true`` but omitted ``data: [DONE]`` gets a single ``[DONE]`` frame.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import signal
from typing import Any, Optional

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.sse_done import DONE_SSE_FRAME, SseDoneTracker, content_type_is_sse

logger = logging.getLogger(__name__)

# Stripped when forwarding upstream: ``host``/``content-length`` are recomputed by aiohttp,
# ``authorization`` is replaced with our bearer; everything else passes through.
_HOP_BY_HOP_HEADERS = frozenset({
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "authorization",
})
# aiohttp recomputes Content-Encoding/Content-Length on stream — let it.
_RESPONSE_DROP_HEADERS = _HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}

DEFAULT_PORT = 8645
DEFAULT_HOST = "127.0.0.1"
# Mirrors api_server's MAX_REQUEST_BYTES (10 MB); client_max_size bounds every read path,
# including chunked bodies.
MAX_REQUEST_BYTES = 10_000_000


def _require_aiohttp() -> None:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for `hermes proxy`. Run `hermes setup` to install it.")


def _json_error(
    status: int,
    message: str,
    code: str = "proxy_error",
    error_type: Optional[str] = None,
) -> "web.Response":
    """OpenAI-style error JSON response."""
    body = {"error": {"message": message, "type": error_type or code, "code": code}}
    return web.json_response(body, status=status)


def _filter_headers(headers, drop: frozenset = _HOP_BY_HOP_HEADERS) -> dict:
    """Strip hop-by-hop (+ auth) headers; ``drop`` widens the set for upstream responses."""
    return {key: value for key, value in headers.items() if key.lower() not in drop}


def _request_peer_is_loopback(request: "web.Request") -> bool:
    """Fail closed unless the transport peer is a numeric loopback address."""
    transport = request.transport
    if transport is None:
        return False
    peer = transport.get_extra_info("peername")
    if not isinstance(peer, (tuple, list)) or not peer:
        return False
    try:
        return ipaddress.ip_address(peer[0]).is_loopback
    except (TypeError, ValueError):
        return False


def _extract_json_body_model(body: bytes) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Return the model hint and parsed object without changing passthrough behavior."""
    if not body:
        return None, None
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    model = parsed.get("model")
    return (model.strip() if isinstance(model, str) and model.strip() else None), parsed


async def _open_upstream(request: "web.Request", rel_path: str, body: bytes, cred: UpstreamCredential):
    """Send the request upstream with ``cred``; returns ``(session, response)`` or
    ``(error_response, None)``."""
    upstream_url = f"{cred.base_url.rstrip('/')}{rel_path}"
    if request.query_string:  # preserved verbatim
        upstream_url = f"{upstream_url}?{request.query_string}"
    fwd_headers = _filter_headers(request.headers)
    fwd_headers["Authorization"] = f"{cred.token_type} {cred.bearer}"
    logger.debug("proxy: forwarding %s %s -> %s (body=%d bytes)", request.method, rel_path, upstream_url, len(body))
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300))
    except Exception as exc:  # pragma: no cover - aiohttp setup issue
        return _json_error(500, f"proxy session init failed: {exc}"), None
    try:
        upstream_resp = await session.request(
            request.method, upstream_url, data=body if body else None, headers=fwd_headers, allow_redirects=False
        )
    except RuntimeError as exc:
        await session.close()
        return _json_error(500, str(exc)), None
    except aiohttp.ClientError as exc:
        await session.close()
        logger.warning("proxy: upstream connection failed: %s", exc)
        return _json_error(502, f"upstream connection failed: {exc}", code="upstream_unreachable"), None
    except asyncio.TimeoutError:
        await session.close()
        return _json_error(504, "upstream request timed out", code="upstream_timeout"), None
    except Exception:
        await session.close()
        raise
    return session, upstream_resp


async def _stream_back(request: "web.Request", session, upstream_resp) -> "web.StreamResponse":
    """Relay status + filtered headers, then the body chunk-by-chunk, appending a missing SSE
    ``[DONE]`` only after a clean EOF."""
    resp = web.StreamResponse(
        status=upstream_resp.status, headers=_filter_headers(upstream_resp.headers, _RESPONSE_DROP_HEADERS)
    )
    await resp.prepare(request)
    done_tracker: Optional[SseDoneTracker] = None
    if content_type_is_sse(upstream_resp.headers):
        done_tracker = SseDoneTracker()
    try:
        async for chunk in upstream_resp.content.iter_any():
            if chunk:
                if done_tracker is not None:
                    done_tracker.feed(chunk)
                await resp.write(chunk)
        if done_tracker is not None and done_tracker.should_append_done():
            try:
                await resp.write(DONE_SSE_FRAME)
            except Exception as exc:  # client hung up at EOF — harmless
                logger.debug("proxy: DONE append skipped: %s", exc)
    except (aiohttp.ClientError, asyncio.CancelledError, OSError) as exc:
        if done_tracker is not None:
            done_tracker.mark_interrupted()
        logger.warning("proxy: streaming interrupted: %s", exc)
    finally:
        upstream_resp.release()
        await session.close()
    await resp.write_eof()
    return resp


def create_app(adapter: UpstreamAdapter) -> "web.Application":
    """Build the aiohttp application bound to a specific upstream adapter.

    Every adapter method is synchronous and blocking (the Nous adapter takes the 15s cross-process
    ``_auth_store_lock()`` and may POST a token refresh; xAI rotates its key pool under a lock),
    so all three are run via ``asyncio.to_thread`` — a contended lock or refresh must never freeze
    the single loop and every other in-flight streaming completion.
    """
    _require_aiohttp()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    image_generation_semaphore = asyncio.Semaphore(1)
    # AppKey: forward-compat with aiohttp versions that strip bare-string keys.
    app[web.AppKey("adapter", UpstreamAdapter)] = adapter

    async def handle_health(request: "web.Request") -> "web.Response":
        authenticated = await asyncio.to_thread(adapter.is_authenticated)
        return web.json_response({"status": "ok", "upstream": adapter.display_name, "authenticated": authenticated})

    async def handle_proxy(request: "web.Request") -> "web.StreamResponse":
        rel_path = "/" + request.match_info.get("tail", "").lstrip("/")
        if rel_path not in adapter.allowed_paths:
            allowed = ", ".join(sorted(adapter.allowed_paths))
            return _json_error(
                404, f"Path /v1{rel_path} is not forwarded by this proxy. Allowed: {allowed}", code="path_not_allowed"
            )
        # Read once and preserve the exact bytes for ordinary forwarding.
        body = await request.read()
        model_hint, parsed_body = _extract_json_body_model(body)

        if rel_path == "/models" and request.method == "GET":
            try:
                aggregated = adapter.list_models()
            except Exception:
                aggregated = None
            if aggregated is not None:
                return web.json_response({"object": "list", "data": aggregated})

        if rel_path == "/chat/completions" and parsed_body is not None:
            raw_route = adapter.raw_chat_route_for_model(model_hint)
            if raw_route is not None:
                try:
                    from gateway.raw_provider import handle_raw_chat_completions

                    return await handle_raw_chat_completions(request, parsed_body, raw_route)
                except Exception as exc:
                    logger.warning("proxy: raw chat route failed: %s", exc)
                    return _json_error(502, "raw chat route failed", code="raw_route_failed")

        if rel_path == "/images/generations":
            if not _request_peer_is_loopback(request):
                return _json_error(
                    403,
                    "Image generation is restricted to loopback clients",
                    code="image_route_forbidden",
                    error_type="permission_error",
                )
            if request.method != "POST":
                response = _json_error(
                    405, "POST is required for /v1/images/generations", code="invalid_request_error"
                )
                response.headers["Allow"] = "POST"
                return response
            if request.content_type != "application/json" or parsed_body is None:
                return _json_error(
                    400,
                    "POST /v1/images/generations requires a JSON object body",
                    code="invalid_request_error",
                )
            image_route = adapter.image_generation_route_for_model(model_hint)
            if image_route is None:
                return _json_error(
                    400,
                    f"No image-generation route is configured for model {model_hint!r}",
                    code="unsupported_image_model",
                )
            try:
                from hermes_cli.proxy.images import handle_image_generation

                return await handle_image_generation(
                    parsed_body, image_route, image_generation_semaphore
                )
            except Exception as exc:
                logger.warning("proxy: image generation route failed: %s", exc)
                return _json_error(502, "image generation route failed", code="image_route_failed")

        try:
            credential_kwargs = {"model": model_hint} if model_hint is not None else {}
            cred = await asyncio.to_thread(adapter.get_credential, **credential_kwargs)
        except Exception as exc:
            logger.warning("proxy: credential resolution failed: %s", exc)
            return _json_error(401, str(exc), code="upstream_auth_failed")
        session, upstream_resp = await _open_upstream(request, rel_path, body, cred)
        if upstream_resp is None:
            return session
        if upstream_resp.status in {401, 429}:
            # One-shot retry with a refreshed/rotated credential (Nous: unconditional refresh
            # POST under the auth lock; xAI: pool rotation).
            try:
                retry_kwargs = {"failed_credential": cred, "status_code": upstream_resp.status}
                if model_hint is not None:
                    retry_kwargs["model"] = model_hint
                retry_cred = await asyncio.to_thread(adapter.get_retry_credential, **retry_kwargs)
            except Exception as exc:
                logger.warning("proxy: retry credential resolution failed: %s", exc)
                retry_cred = None
            if retry_cred is not None:
                upstream_resp.release()
                await session.close()
                session, upstream_resp = await _open_upstream(request, rel_path, body, retry_cred)
                if upstream_resp is None:
                    return session
        return await _stream_back(request, session, upstream_resp)

    app.router.add_get("/health", handle_health)  # never goes upstream
    app.router.add_route("*", "/v1/{tail:.*}", handle_proxy)  # forwards if the path is allowed
    return app


async def run_server(
    adapter: UpstreamAdapter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the proxy in the current event loop until shutdown_event is set."""
    _require_aiohttp()
    app = create_app(adapter)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("proxy: listening on http://%s:%d/v1 -> %s", host, port, adapter.display_name)
    stop_event = shutdown_event or asyncio.Event()
    if shutdown_event is None:  # we own the loop's lifetime → wire signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)  # windows-footgun: ok
            except NotImplementedError:
                pass  # Windows / restricted envs — Ctrl+C still raises KeyboardInterrupt
    try:
        await stop_event.wait()
    finally:
        logger.info("proxy: shutting down")
        await runner.cleanup()


__all__ = ["create_app", "run_server", "DEFAULT_HOST", "DEFAULT_PORT", "AIOHTTP_AVAILABLE"]
