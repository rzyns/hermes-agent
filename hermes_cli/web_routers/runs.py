"""Dashboard proxy routes for api-server run events."""

from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from hermes_cli.web_deps import late


router = APIRouter()

_require_token = late("_require_token")
_require_token_or_query = late("_require_token_or_query")
_facade_resolve_api_server_base_url = late("_resolve_api_server_base_url")
_facade_resolve_api_server_key = late("_resolve_api_server_key")
_facade_run_events_unavailable_bytes = late("_run_events_unavailable_bytes")
_facade_run_events_unavailable_frame = late("_run_events_unavailable_frame")

_RUN_EVENTS_UPSTREAM_CONNECT_TIMEOUT = 3.0
_RUN_EVENTS_UPSTREAM_READ_TIMEOUT: Optional[float] = None


def _resolve_api_server_base_url() -> str:
    """Resolve the api-server URL from env, profile config, then defaults."""
    host = os.getenv("API_SERVER_HOST", "")
    port_raw = os.getenv("API_SERVER_PORT", "")
    if not host or not port_raw:
        try:
            from hermes_cli.config import read_user_config_raw

            cfg = read_user_config_raw()
            gateway = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
            gateway_api = (
                gateway.get("api_server")
                if isinstance(gateway.get("api_server"), dict)
                else {}
            )
            platforms = cfg.get("platforms") if isinstance(cfg.get("platforms"), dict) else {}
            platform_api = (
                platforms.get("api_server")
                if isinstance(platforms.get("api_server"), dict)
                else {}
            )
            merged: Dict[str, Any] = {**gateway_api, **platform_api}
            host = host or str(merged.get("host", ""))
            port_raw = port_raw or str(merged.get("port", ""))
        except Exception:
            pass
    host = host or "127.0.0.1"
    try:
        port = int(port_raw or "8642")
    except (TypeError, ValueError):
        port = 8642
    return f"http://{host}:{port}"


def _resolve_api_server_key() -> str:
    """Resolve the profile-scoped api-server credential."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            return get_secret("API_SERVER_KEY", "") or ""
        except UnscopedSecretError:
            return os.getenv("API_SERVER_KEY", "")
    except Exception:
        return os.getenv("API_SERVER_KEY", "")


def _run_events_unavailable_bytes(reason: str) -> bytes:
    payload = json.dumps({"proxy_error": True, "reason": reason})
    return (
        f"event: hermes.run_events.proxy_error\ndata: {payload}\n\n"
        ": stream closed\n\n"
    ).encode()


def _run_events_unavailable_frame(reason: str) -> StreamingResponse:
    async def _single():
        yield _run_events_unavailable_bytes(reason)

    return StreamingResponse(
        _single(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/runs/{run_id}/events")
async def proxy_run_events(request: Request, run_id: str):
    """Forward an authenticated SSE stream from the gateway api server."""
    _require_token_or_query(request)
    try:
        import httpx
    except ImportError:
        return _facade_run_events_unavailable_frame("httpx is not available")

    base = _facade_resolve_api_server_base_url()
    api_key = _facade_resolve_api_server_key()
    upstream_url = (
        f"{base}/v1/runs/{urllib.parse.quote(run_id, safe='')}/events"
    )
    params = {
        key: value
        for key in ("include", "after")
        if (value := request.query_params.get(key)) is not None
    }
    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if last_event_id := request.headers.get("Last-Event-ID"):
        headers["Last-Event-ID"] = last_event_id

    async def _stream():
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=_RUN_EVENTS_UPSTREAM_CONNECT_TIMEOUT,
                    read=_RUN_EVENTS_UPSTREAM_READ_TIMEOUT,
                    write=10.0,
                    pool=5.0,
                )
            ) as client:
                async with client.stream(
                    "GET", upstream_url, params=params, headers=headers
                ) as upstream:
                    if upstream.status_code != 200:
                        body = await upstream.aread()
                        payload = json.dumps(
                            {
                                "proxy_error": True,
                                "upstream_status": upstream.status_code,
                                "upstream_body": body.decode("utf-8", "replace")[:1000],
                            }
                        )
                        yield (
                            "event: hermes.run_events.proxy_error\n"
                            f"data: {payload}\n\n: stream closed\n\n"
                        ).encode()
                        return
                    async for chunk in upstream.aiter_raw():
                        if chunk:
                            yield chunk
        except httpx.ConnectError:
            yield _facade_run_events_unavailable_bytes("api_server is not reachable")
        except httpx.ReadTimeout:
            yield _facade_run_events_unavailable_bytes("api_server read timed out")
        except Exception as exc:
            yield _facade_run_events_unavailable_bytes(
                f"proxy error: {type(exc).__name__}"
            )

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/api/runs/capabilities")
async def get_run_events_capabilities(request: Request):
    """Return the gateway api server's advertised run-events capability."""
    _require_token(request)
    try:
        import httpx
    except ImportError:
        return JSONResponse({"available": False, "reason": "httpx is not available"})

    base = _facade_resolve_api_server_base_url()
    headers: Dict[str, str] = {}
    if api_key := _facade_resolve_api_server_key():
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(_RUN_EVENTS_UPSTREAM_CONNECT_TIMEOUT)
        ) as client:
            response = await client.get(f"{base}/v1/capabilities", headers=headers)
            if response.status_code != 200:
                return JSONResponse(
                    {
                        "available": False,
                        "reason": f"upstream status {response.status_code}",
                    },
                    status_code=502,
                )
            data = response.json() or {}
            return JSONResponse(
                {
                    "available": bool((data.get("features") or {}).get("run_events_sse")),
                    "run_events": data.get("run_events") or {},
                    "endpoint": (data.get("endpoints") or {}).get("run_events"),
                }
            )
    except httpx.ConnectError:
        return JSONResponse(
            {"available": False, "reason": "api_server is not reachable"}
        )
    except Exception as exc:
        return JSONResponse(
            {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        )