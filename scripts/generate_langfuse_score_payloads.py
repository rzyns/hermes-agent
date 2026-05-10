#!/usr/bin/env python3
"""Materialize conservative Langfuse score payloads from local eval artifacts.

This script is intentionally no-write. It prepares inspectable payloads for a
future explicit score-write gate after a dataset experiment/run has been created.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib import error, request


class ScorePayloadError(ValueError):
    """Raised when score payload inputs are unsafe or ambiguous."""


SCORE_ALLOWLIST: dict[str, dict[str, Any]] = {
    "privacy_safe": {
        "dataType": "BOOLEAN",
        "value": 1,
        "scope": "dataset_run",
        "endpoint": "/api/public/scores",
    },
    "test_passed": {
        "dataType": "BOOLEAN",
        "value": 1,
        "scope": "dataset_run",
        "endpoint": "/api/public/scores",
    },
    "task_success": {
        "dataType": "BOOLEAN",
        "value": 1,
        "scope": "dataset_run",
        "endpoint": "/api/public/scores",
    },
}
DUPLICATE_POLICY = "fail_closed_on_multiple_scores_per_dataset_run_name"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _single_dataset_run_id(write_report: Mapping[str, Any]) -> str:
    langfuse_write = write_report.get("langfuse_write")
    if not isinstance(langfuse_write, Mapping):
        raise ScorePayloadError("langfuse write artifact is missing langfuse_write envelope")
    if int(langfuse_write.get("failed_run_item_count", 0) or 0) != 0:
        raise ScorePayloadError("langfuse write artifact contains failed run item writes")
    ids = {
        str(item.get("dataset_run_id"))
        for item in langfuse_write.get("items", [])
        if isinstance(item, Mapping) and item.get("dataset_run_id")
    }
    if len(ids) != 1:
        raise ScorePayloadError("expected exactly one dataset run id in langfuse write artifact")
    return next(iter(ids))


def _semantic_summary(semantic_adjudication: Mapping[str, Any] | None) -> dict[str, int] | None:
    if semantic_adjudication is None:
        return None
    summary = semantic_adjudication.get("summary") if isinstance(semantic_adjudication.get("summary"), Mapping) else {}
    return {
        "manual_check_count": int(summary.get("manual_check_count", 0) or 0),
        "pass_count": int(summary.get("pass_count", 0) or 0),
        "fail_count": int(summary.get("fail_count", 0) or 0),
        "unclear_count": int(summary.get("unclear_count", 0) or 0),
        "missing_adjudication_count": int(summary.get("missing_adjudication_count", 0) or 0),
    }


def _semantic_scores_are_eligible(semantic_adjudication: Mapping[str, Any] | None) -> bool:
    semantic = _semantic_summary(semantic_adjudication)
    if semantic is None:
        return False
    if semantic["manual_check_count"] <= 0:
        return False
    if semantic["fail_count"] or semantic["unclear_count"] or semantic["missing_adjudication_count"]:
        return False
    policy = semantic_adjudication.get("score_policy_v1") if isinstance(semantic_adjudication.get("score_policy_v1"), Mapping) else {}
    return (
        policy.get("test_passed") == "eligible_for_future_write_gate"
        and policy.get("task_success") == "eligible_for_future_write_gate"
    )


def validate_score_payloads_against_allowlist(score_payloads: Mapping[str, Any]) -> None:
    """Fail closed unless every materialized score matches the per-score allowlist."""
    dataset_run_id = score_payloads.get("dataset_run_id")
    payloads = score_payloads.get("score_payloads")
    if not dataset_run_id:
        raise ScorePayloadError("score payload envelope missing dataset_run_id")
    if not isinstance(payloads, list):
        raise ScorePayloadError("score_payloads must be a list")
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ScorePayloadError("score payload entry must be an object")
        endpoint = payload.get("endpoint")
        body = payload.get("body")
        if not isinstance(body, Mapping):
            raise ScorePayloadError("score payload missing body")
        name = str(body.get("name") or "")
        allowed = SCORE_ALLOWLIST.get(name)
        if allowed is None:
            raise ScorePayloadError(f"score {name!r} is not in the per-score allowlist")
        checks = {
            "endpoint": endpoint,
            "dataType": body.get("dataType"),
            "value": body.get("value"),
            "scope": body.get("metadata", {}).get("scope") if isinstance(body.get("metadata"), Mapping) else None,
        }
        for field, observed in checks.items():
            if observed != allowed[field]:
                raise ScorePayloadError(f"allowlist mismatch for {name}.{field}: expected {allowed[field]!r}, got {observed!r}")
        if body.get("datasetRunId") != dataset_run_id:
            raise ScorePayloadError("score payload datasetRunId does not match envelope")


def _score_payload_bodies(score_payloads: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    validate_score_payloads_against_allowlist(score_payloads)
    return [payload["body"] for payload in score_payloads.get("score_payloads", [])]


def _extract_readback_scores(readback_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Accept common Langfuse list response shapes without requiring live API access."""
    candidates = readback_report.get("data")
    if candidates is None:
        candidates = readback_report.get("scores")
    if candidates is None and isinstance(readback_report.get("result"), Mapping):
        candidates = readback_report["result"].get("data")
    if not isinstance(candidates, list):
        raise ScorePayloadError("score readback report must contain a list under data or scores")
    return [score for score in candidates if isinstance(score, Mapping)]


def _write_response_ids(write_report: Mapping[str, Any] | None) -> dict[str, str | None]:
    if not write_report:
        return {}
    scores = write_report.get("scores")
    if not isinstance(scores, list):
        return {}
    ids: dict[str, str | None] = {}
    for score in scores:
        if isinstance(score, Mapping) and score.get("name"):
            response_id = score.get("response_id")
            ids[str(score["name"])] = str(response_id) if response_id else None
    return ids


def verify_score_readback(
    score_payloads: Mapping[str, Any],
    write_report: Mapping[str, Any] | None,
    readback_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify dataset-run filtered score list readback against intended payloads.

    Direct score-id GET may be unavailable for semantic score response ids, so this
    verifier consumes a list response filtered by datasetRunId and then checks name,
    dataType, value, scope, and optional write response ids locally.
    """
    expected_bodies = _score_payload_bodies(score_payloads)
    dataset_run_id = str(score_payloads.get("dataset_run_id") or "")
    readback_scores = [
        score
        for score in _extract_readback_scores(readback_report)
        if str(score.get("datasetRunId") or score.get("dataset_run_id") or "") == dataset_run_id
    ]
    response_ids_by_name = _write_response_ids(write_report)
    results: list[dict[str, Any]] = []
    matched_count = 0
    overall_status = "passed"
    for body in expected_bodies:
        name = str(body.get("name") or "")
        matches = [score for score in readback_scores if score.get("name") == name]
        duplicate_status = "none" if len(matches) <= 1 else "duplicate_dataset_run_score"
        expected_response_id = response_ids_by_name.get(name)
        matched_response_ids = [str(score.get("id")) for score in matches if score.get("id")]
        status = "passed"
        if len(matches) != 1:
            status = "failed"
        else:
            match = matches[0]
            if match.get("dataType") != body.get("dataType") or match.get("value") != body.get("value"):
                status = "failed"
            if expected_response_id and expected_response_id not in matched_response_ids:
                status = "failed"
        if status == "passed":
            matched_count += 1
        else:
            overall_status = "failed"
        results.append({
            "name": name,
            "status": status,
            "expected_response_id": expected_response_id,
            "matched_response_ids": matched_response_ids,
            "duplicate_status": duplicate_status,
        })
    return {
        "mode": "langfuse_score_readback_verification",
        "dataset_run_id": dataset_run_id,
        "status": overall_status,
        "matched_score_count": matched_count,
        "expected_score_count": len(expected_bodies),
        "duplicate_policy": DUPLICATE_POLICY,
        "readback_filter": {"datasetRunId": dataset_run_id},
        "scores": results,
    }


def build_score_payloads(
    local_eval: Mapping[str, Any],
    write_report: Mapping[str, Any],
    semantic_adjudication: Mapping[str, Any] | None = None,
    *,
    include_privacy_score: bool = True,
) -> dict[str, Any]:
    dataset_name = str(local_eval.get("dataset_name") or "")
    if dataset_name != "hermes/turn-regression/pilot":
        raise ScorePayloadError("refusing score payloads outside hermes/turn-regression/pilot")
    langfuse_write = write_report.get("langfuse_write")
    if not isinstance(langfuse_write, Mapping):
        raise ScorePayloadError("langfuse write artifact is missing langfuse_write envelope")
    if langfuse_write.get("dataset_name") != dataset_name:
        raise ScorePayloadError("local eval dataset and langfuse write dataset do not match")
    summary = local_eval.get("summary") if isinstance(local_eval.get("summary"), Mapping) else {}
    local_fail_count = int(summary.get("fail_count", 0) or 0)
    secret_findings = int(summary.get("secret_findings", 0) or 0)
    manual_pending_count = int(summary.get("pending_manual_count", 0) or 0)
    evaluated_count = int(summary.get("evaluated_count", 0) or 0)
    if local_fail_count:
        raise ScorePayloadError("local eval has failures; refusing score payload materialization")
    dataset_run_id = _single_dataset_run_id(write_report)
    run_name = str(langfuse_write.get("run_name") or "")

    payloads: list[dict[str, Any]] = []
    if secret_findings == 0 and include_privacy_score:
        payloads.append({
            "endpoint": "/api/public/scores",
            "body": {
                "name": "privacy_safe",
                "value": 1,
                "dataType": "BOOLEAN",
                "datasetRunId": dataset_run_id,
                "comment": (
                    f"Hermes local evaluator: 0 secret findings across {evaluated_count} "
                    "evaluated pilot items. Manual semantic checks remain pending."
                ),
                "metadata": {
                    "dataset_name": dataset_name,
                    "run_name": run_name,
                    "producer": "scripts/generate_langfuse_score_payloads.py",
                    "scope": "dataset_run",
                    "manual_pending_count": manual_pending_count,
                },
            },
        })

    semantic = _semantic_summary(semantic_adjudication)
    semantic_eligible = _semantic_scores_are_eligible(semantic_adjudication)
    if semantic_eligible and semantic is not None:
        semantic_comment = (
            f"Hermes semantic adjudication: {semantic['pass_count']}/{semantic['manual_check_count']} manual checks passed "
            f"with {semantic['fail_count']} failures and {semantic['unclear_count']} unclear checks."
        )
        for score_name in ("test_passed", "task_success"):
            payloads.append({
                "endpoint": "/api/public/scores",
                "body": {
                    "name": score_name,
                    "value": 1,
                    "dataType": "BOOLEAN",
                    "datasetRunId": dataset_run_id,
                    "comment": semantic_comment,
                    "metadata": {
                        "dataset_name": dataset_name,
                        "run_name": run_name,
                        "producer": "scripts/generate_langfuse_score_payloads.py",
                        "scope": "dataset_run",
                        "semantic_adjudication_status": "fully_passing",
                        "semantic_manual_check_count": semantic["manual_check_count"],
                        "semantic_pass_count": semantic["pass_count"],
                        "semantic_fail_count": semantic["fail_count"],
                        "semantic_unclear_count": semantic["unclear_count"],
                    },
                },
            })

    if semantic_eligible:
        deferred_scores: list[dict[str, Any]] = []
    else:
        defer_reason = "manual semantic checks remain pending; do not mark run as test-passed yet"
        deferred_count = manual_pending_count
        if semantic is not None:
            defer_reason = "semantic adjudication is not fully passing; do not mark run as test-passed yet"
            deferred_count = semantic["fail_count"] + semantic["unclear_count"] + semantic["missing_adjudication_count"]
        deferred_scores = [
            {
                "score_name": "test_passed",
                "reason": defer_reason,
                "deferred_check_count": deferred_count,
            },
            {
                "score_name": "task_success",
                "reason": "pilot replay outputs are review aids, not final acceptance outcomes" if semantic is None else defer_reason,
                "deferred_check_count": deferred_count,
            },
        ]
    result = {
        "mode": "langfuse_score_payloads_no_write",
        "write_enabled": False,
        "dataset_name": dataset_name,
        "run_name": run_name,
        "dataset_run_id": dataset_run_id,
        "score_allowlist": SCORE_ALLOWLIST,
        "duplicate_policy": DUPLICATE_POLICY,
        "separated_write_surface": {
            "materialization": "default_no_write_payload_generation",
            "execution": "requires --write-scores and --confirm-score-write",
            "readback": "verify with dataset-run filtered score list",
        },
        "summary": {
            "payload_count": len(payloads),
            "deferred_score_count": len(deferred_scores),
            "manual_pending_count": manual_pending_count,
            "secret_findings": secret_findings,
            "local_fail_count": local_fail_count,
        },
        "score_payloads": payloads,
        "deferred_scores": deferred_scores,
        "requires_explicit_future_flags": ["--write-scores", "--confirm-score-write"],
    }
    if semantic is not None:
        result["summary"].update({
            "semantic_pass_count": semantic["pass_count"],
            "semantic_fail_count": semantic["fail_count"],
            "semantic_unclear_count": semantic["unclear_count"],
        })
    return result


PostScore = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _langfuse_credentials() -> tuple[str, str, str]:
    _load_env_file(Path.home() / ".hermes" / ".env")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("HERMES_LANGFUSE_BASE_URL")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or os.getenv("HERMES_LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("HERMES_LANGFUSE_SECRET_KEY")
    missing = [
        name
        for name, value in (
            ("LANGFUSE_HOST/HERMES_LANGFUSE_BASE_URL", host),
            ("LANGFUSE_PUBLIC_KEY/HERMES_LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY/HERMES_LANGFUSE_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise ScorePayloadError("missing Langfuse credential fields: " + ", ".join(missing))
    return host.rstrip("/"), public_key, secret_key


def _post_langfuse_score(body: Mapping[str, Any]) -> Mapping[str, Any]:
    host, public_key, secret_key = _langfuse_credentials()
    basic_auth_value = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        f"{host}/api/public/scores",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Basic {basic_auth_value}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:  # noqa: S310 - approved Langfuse endpoint
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {"status": resp.status}
    except error.HTTPError as exc:  # pragma: no cover - live-network guard
        body_text = exc.read().decode("utf-8", errors="replace")[:500]
        raise ScorePayloadError(f"Langfuse score write failed HTTP {exc.code}: {body_text}") from exc
    except error.URLError as exc:  # pragma: no cover - live-network guard
        raise ScorePayloadError(f"Langfuse score write failed: {exc.reason}") from exc


def write_score_payloads(
    score_payloads: Mapping[str, Any],
    *,
    confirm_score_write: bool,
    post_score: PostScore | None = None,
) -> dict[str, Any]:
    if not confirm_score_write:
        raise ScorePayloadError("score writes require --confirm-score-write")
    if score_payloads.get("dataset_name") != "hermes/turn-regression/pilot":
        raise ScorePayloadError("refusing score writes outside hermes/turn-regression/pilot")
    payloads = score_payloads.get("score_payloads")
    if not isinstance(payloads, list):
        raise ScorePayloadError("score_payloads must be a list")
    validate_score_payloads_against_allowlist(score_payloads)
    bodies: list[Mapping[str, Any]] = []
    for payload in payloads:
        body = payload["body"]
        bodies.append(body)

    names = [str(body.get("name") or "") for body in bodies]
    approved_single_privacy = names == ["privacy_safe"]
    approved_semantic_pair = names == ["test_passed", "task_success"]
    if not (approved_single_privacy or approved_semantic_pair):
        raise ScorePayloadError("expected either the single privacy_safe score or exact semantic score pair")
    if approved_semantic_pair:
        summary = score_payloads.get("summary") if isinstance(score_payloads.get("summary"), Mapping) else {}
        if summary.get("semantic_fail_count") != 0 or summary.get("semantic_unclear_count") != 0:
            raise ScorePayloadError("semantic score writes require fully passing adjudication summary")

    writer = post_score or _post_langfuse_score
    written_scores: list[dict[str, Any]] = []
    for body in bodies:
        response = writer(body)
        response_id = response.get("id") if isinstance(response, Mapping) else None
        written_scores.append({
            "name": body.get("name"),
            "dataset_run_id": body.get("datasetRunId"),
            "write_status": "created",
            "response_id": str(response_id) if response_id else None,
        })
    return {
        "mode": "langfuse_score_write",
        "write_enabled": True,
        "dataset_name": score_payloads.get("dataset_name"),
        "run_name": score_payloads.get("run_name"),
        "dataset_run_id": score_payloads.get("dataset_run_id"),
        "created_score_count": len(written_scores),
        "failed_score_count": 0,
        "scores": written_scores,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate conservative Langfuse score payloads, optionally writing after explicit approval.")
    parser.add_argument("--local-eval-json", type=Path, required=True)
    parser.add_argument("--langfuse-write-json", type=Path, required=True)
    parser.add_argument("--semantic-adjudication-json", type=Path, help="Optional no-write semantic adjudication artifact for test_passed/task_success payloads")
    parser.add_argument("--semantic-scores-only", action="store_true", help="Materialize only eligible test_passed/task_success payloads; omit privacy_safe if it was already written")
    parser.add_argument("--output-json", type=Path, help="Optional output path")
    parser.add_argument("--write-scores", action="store_true", help="Write conservative score payloads to Langfuse")
    parser.add_argument("--confirm-score-write", action="store_true", help="Required with --write-scores")
    parser.add_argument("--score-write-result-json", type=Path, help="Optional score write result artifact with response ids for readback verification")
    parser.add_argument("--score-readback-json", type=Path, help="Optional dataset-run filtered score list readback artifact to verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        semantic_adjudication = _load_json(args.semantic_adjudication_json) if args.semantic_adjudication_json else None
        if args.semantic_scores_only and semantic_adjudication is None:
            raise ScorePayloadError("--semantic-scores-only requires --semantic-adjudication-json")
        materialized_result = build_score_payloads(
            _load_json(args.local_eval_json),
            _load_json(args.langfuse_write_json),
            semantic_adjudication,
            include_privacy_score=not args.semantic_scores_only,
        )
        result = materialized_result
        if args.write_scores:
            result = write_score_payloads(materialized_result, confirm_score_write=args.confirm_score_write)
        elif args.confirm_score_write:
            raise ScorePayloadError("--confirm-score-write requires --write-scores")
        if args.score_readback_json:
            write_result_artifact = _load_json(args.score_write_result_json) if args.score_write_result_json else (result if args.write_scores else None)
            result = verify_score_readback(materialized_result, write_result_artifact, _load_json(args.score_readback_json))
        elif args.score_write_result_json:
            raise ScorePayloadError("--score-write-result-json requires --score-readback-json")
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered + "\n")
        print(rendered)
        return 0
    except ScorePayloadError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
