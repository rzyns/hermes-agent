from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "score_langfuse_turn.py"


def load_script():
    spec = importlib.util.spec_from_file_location("score_langfuse_turn", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_no_write_by_default(capsys):
    mod = load_script()
    rc = mod.main(["--trace-id", "trace_12345678", "--comment", "safe comment"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["write_requested"] is False
    assert out["wrote_score"] is False
    assert out["payload"]["traceId"] == "trace_12345678"
    assert out["payload"]["dataType"] == "BOOLEAN"
    assert out["payload"]["value"] == 1


def test_write_requires_second_confirmation():
    mod = load_script()
    with pytest.raises(SystemExit, match="confirm"):
        mod.main(["--trace-id", "trace_12345678", "--write-score"])


def test_unallowlisted_score_name_fails_closed():
    mod = load_script()
    with pytest.raises(SystemExit, match="not allowlisted"):
        mod.build_score_payload("trace_12345678", "surprise_score", "comment")


def test_comment_redaction_removes_secret_like_values():
    mod = load_script()
    payload = mod.build_score_payload(
        "trace_12345678",
        "lf11_report_only_tool_trace_score_write_success",
        "token=abc123 bearer secret-token pk-lf-public",
    )
    comment = payload["comment"].lower()
    assert "abc123" not in comment
    assert "secret-token" not in comment
    assert "pk-lf-public" not in comment
    assert "<redacted>" in comment


def test_write_path_uses_returned_score_id_and_trace_get(monkeypatch, capsys):
    mod = load_script()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.test")
    calls = []

    def fake_api_request(host, public_key, secret_key, method, path, payload=None):
        calls.append((method, path, payload))
        if method == "POST" and path == "/api/public/scores":
            return mod.ApiResult(200, {"id": "score_1"})
        if method == "GET" and path == "/api/public/scores/score_1":
            return mod.ApiResult(200, {"id": "score_1"})
        if method == "GET" and path == "/api/public/traces/trace_12345678":
            return mod.ApiResult(200, {"scores": [{"id": "score_1", "name": "lf11_report_only_tool_trace_score_write_success", "dataType": "BOOLEAN", "value": 1}]})
        raise AssertionError((method, path))

    monkeypatch.setattr(mod, "api_request", fake_api_request)
    rc = mod.main(["--trace-id", "trace_12345678", "--write-score", "--confirm-score-write"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["wrote_score"] is True
    assert out["score_id"] == "score_1"
    assert [c[0:2] for c in calls] == [
        ("POST", "/api/public/scores"),
        ("GET", "/api/public/scores/score_1"),
        ("GET", "/api/public/traces/trace_12345678"),
    ]
