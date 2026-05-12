#!/usr/bin/env python3
"""Conservative one-turn Langfuse trace score helper.

Writes are disabled by default. LF11 live validation may enable exactly one
report-only BOOLEAN score after review by passing both --write-score and
--confirm-score-write.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

ALLOWED_SCORE_NAMES = {
    "lf11_report_only_tool_trace_score_write_success",
    "score_write_success",
}
_SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(pk|sk)-lf-[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+"),
]


@dataclass
class ApiResult:
    status: int
    body: Any


def redact_text(value: str, *, max_chars: int = 300) -> str:
    text = value or ""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "<redacted>", text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"
    return text


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def resolve_config(args: argparse.Namespace) -> tuple[str, str, str]:
    host = args.host or _env("LANGFUSE_HOST") or _env("HERMES_LANGFUSE_BASE_URL")
    public_key = _env("LANGFUSE_PUBLIC_KEY") or _env("HERMES_LANGFUSE_PUBLIC_KEY")
    secret_key = _env("LANGFUSE_SECRET_KEY") or _env("HERMES_LANGFUSE_SECRET_KEY")
    if not host:
        raise SystemExit("missing Langfuse host (set LANGFUSE_HOST or HERMES_LANGFUSE_BASE_URL)")
    if not (public_key and secret_key):
        raise SystemExit("missing Langfuse API keys in standard or HERMES_LANGFUSE_* env vars")
    return host.rstrip("/"), public_key, secret_key


def api_request(host: str, public_key: str, secret_key: str, method: str, path: str, payload: dict[str, Any] | None = None) -> ApiResult:
    url = host + path
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hermes-lf11-score-helper/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return ApiResult(resp.status, json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            body: Any = json.loads(raw) if raw else {}
        except Exception:
            body = {"error": redact_text(raw)}
        return ApiResult(exc.code, body)


def build_score_payload(trace_id: str, name: str, comment: str) -> dict[str, Any]:
    if not trace_id or not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", trace_id):
        raise SystemExit("trace id must be one explicit safe identifier")
    if name not in ALLOWED_SCORE_NAMES:
        raise SystemExit(f"score name not allowlisted: {name}")
    return {
        "traceId": trace_id,
        "name": name,
        "value": 1,
        "dataType": "BOOLEAN",
        "comment": redact_text(comment, max_chars=300),
    }


def score_id_from_body(body: Any) -> str:
    if isinstance(body, dict):
        for key in ("id", "scoreId", "score_id"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
        data = body.get("data")
        if isinstance(data, dict):
            return score_id_from_body(data)
    return ""


def trace_has_score(trace_body: Any, score_id: str, score_name: str) -> bool:
    if not isinstance(trace_body, dict):
        return False
    candidates = []
    for key in ("scores", "score", "data"):
        value = trace_body.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif isinstance(value, dict) and isinstance(value.get("scores"), list):
            candidates.extend(value["scores"])
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if score_id and item.get("id") == score_id:
            return True
        if item.get("name") == score_name and item.get("dataType") == "BOOLEAN" and item.get("value") in (1, 1.0, True):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-id", required=True, help="single controlled Langfuse trace id")
    parser.add_argument("--name", default="lf11_report_only_tool_trace_score_write_success")
    parser.add_argument("--comment", default="LF11 report-only tool trace score write/readback smoke")
    parser.add_argument("--host", default="")
    parser.add_argument("--write-score", action="store_true", help="actually create the score")
    parser.add_argument("--confirm-score-write", action="store_true", help="second live-write confirmation")
    args = parser.parse_args(argv)

    payload = build_score_payload(args.trace_id, args.name, args.comment)
    result: dict[str, Any] = {
        "trace_id": args.trace_id,
        "score_name": args.name,
        "payload": payload,
        "write_requested": bool(args.write_score),
        "write_confirmed": bool(args.confirm_score_write),
        "wrote_score": False,
        "self_hosted_scores_traceid_caveat": "Do not rely solely on GET /api/public/scores?traceId for self-hosted readback; verify returned score id plus trace-get evidence.",
    }

    if not args.write_score:
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.confirm_score_write:
        raise SystemExit("live score write requires --confirm-score-write")

    host, public_key, secret_key = resolve_config(args)
    create = api_request(host, public_key, secret_key, "POST", "/api/public/scores", payload)
    result["create_status"] = create.status
    result["create_ok"] = 200 <= create.status < 300
    score_id = score_id_from_body(create.body)
    result["score_id"] = score_id
    if not result["create_ok"] or not score_id:
        result["error"] = redact_text(json.dumps(create.body, sort_keys=True))
        print(json.dumps(result, sort_keys=True))
        return 2

    # Returned score-id readback plus trace-get verification.  Both paths are
    # deliberately narrow and report-only for exactly one controlled trace.
    score_get = api_request(host, public_key, secret_key, "GET", f"/api/public/scores/{urllib.parse.quote(score_id)}")
    result["score_get_status"] = score_get.status
    result["score_get_ok"] = 200 <= score_get.status < 300
    trace_get = api_request(host, public_key, secret_key, "GET", f"/api/public/traces/{urllib.parse.quote(args.trace_id)}")
    result["trace_get_status"] = trace_get.status
    result["trace_get_ok"] = 200 <= trace_get.status < 300
    result["trace_get_score_linked"] = trace_has_score(trace_get.body, score_id, args.name)
    result["wrote_score"] = bool(result["score_get_ok"] and result["trace_get_ok"] and result["trace_get_score_linked"])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["wrote_score"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
