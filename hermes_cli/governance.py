"""Governance evaluation — dry-run/report-only CLI surface (Surface A).

Conceptually equivalent to ``hermes governance evaluate --dry-run``.
Local-only: reads explicit input fixtures, writes decision artifacts,
does not mutate Kanban/Dashboard/Gateway state.

Surface A scope and limitations:
- Evaluates local JSON fixtures only.  No live Kanban/Dashboard/Gateway
  integration, no freshness/identity verification, no manifest collector
  provenance beyond the in-band collector identity string.  Stale-input
  semantics and full manifest-schema identity are deferred to Surface B.
- The minimal valid schema is a JSON object containing at least a known
  ``status`` field.  Any missing, non-object, or structurally incomplete
  input is rejected fail-closed.

Exit-code contract (M3 addendum):
    0  — evaluation completed; decision artifact written.
           Policy result may be allow or block; parse artifact for meaning.
    10 — policy blocked; decision artifact written.
    20 — no authoritative decision artifact written;
           fail-closed validation/runtime condition (malformed input,
           missing required field, unknown vocabulary, type mismatch).
    30 — unexpected tool/runtime failure; no authoritative decision artifact.

Invariant: exit code 0/10 guarantee decision artifact exists.
           exit code 20/30 guarantee no authoritative decision artifact exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTOR_IDENTITY = "hgk-governance-collector:v1"

# Frozen source-derived vocabulary from kanban_db.py (M3 addendum Repair 3).
# Task table VALID_STATUSES — exact set from hermes_cli/kanban_db.py line 97.
VALID_STATUSES = {
    "triage",
    "todo",
    "scheduled",
    "ready",
    "running",
    "blocked",
    "review",
    "done",
    "archived",
}

# Reconciled run vocabulary from kanban_db.py schema comment / transition literals.
# The set includes None while running, plus terminal outcomes produced by
# _end_run / _synthesize_ended_run as observed during M3 repair.
# Excluded values: operator_cancelled, success, closed (not source-derived).
VALID_OUTCOMES = {
    None,
    "completed",
    "blocked",
    "crashed",
    "timed_out",
    "spawn_failed",
    "reclaimed",
    "gave_up",
    "scheduled",
    "stale",
}

# Values explicitly not accepted unless a future implementation cites a
# producing source and updates the frozen vocabulary table above.
UNSUPPORTED_VALUES = {"operator_cancelled", "success", "closed"}

EXIT_OK = 0
EXIT_BLOCKED = 10
EXIT_FAIL_CLOSED = 20
EXIT_RUNTIME_ERROR = 30


# ---------------------------------------------------------------------------
# Surface B — local manifest/redaction validator (non-collector, non-mutation)
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_FIELDS = {
    "manifest_version",
    "schema_identity",
    "bundle_id",
    "task_id",
    "run_id",
    "task_status",
    "run_status",
    "run_outcome",
    "workspace_identity",
    "artifact_refs",
    "collector_identity",
    "collection_started_at",
    "collection_finished_at",
    "source_vocabulary",
    "redaction_state",
    "authority_class",
    "non_authorizations",
    "board_identity",
}

ALLOWED_REDACTION_STATES = {
    "raw_only",
    "redacted",
    "redaction_failed",
    "not_exportable",
}

ALLOWED_AUTHORITY_CLASSES = {
    "diagnostic",
    "candidate_decision",
    "authoritative_decision",
    "export_summary",
}

SURFACE_B_EXIT_CODES = {
    "EXIT_OK": 0,
    "EXIT_VALIDATION_FAILED": 10,
    "EXIT_SCHEMA_MISMATCH": 20,
    "EXIT_RUNTIME_ERROR": 30,
}

# Canonicalization hardening constants
_MAX_MANIFEST_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB
_MAX_MANIFEST_DEPTH = 32  # exclusive; depth 0 is root dict
_KNOWN_MANIFEST_VERSIONS = {"hgk.surface_b.manifest.v1"}


def build_parser(subparsers: Any) -> Any:
    """Attach ``governance`` parser to the shared subparsers object."""
    gov_parser = subparsers.add_parser(
        "governance",
        help="Governance evaluation (dry-run/report-only)",
        description=(
            "Local governance evaluation surface. Reads local input fixtures "
            "and writes decision artifacts. Does not mutate Kanban/Dashboard/"
            "Gateway state or trigger notifications."
        ),
    )
    gov_subparsers = gov_parser.add_subparsers(dest="governance_command")

    evaluate_parser = gov_subparsers.add_parser(
        "evaluate",
        help="Evaluate a local governance input fixture",
    )
    evaluate_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Enable dry-run/report-only mode (required).",
    )
    evaluate_parser.add_argument(
        "--input",
        dest="input",
        required=True,
        help="Path to the input fixture (JSON snapshot or manifest).",
    )
    evaluate_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Directory where the decision artifact will be written.",
    )
    evaluate_parser.add_argument(
        "--format",
        dest="format",
        default="json",
        choices=["json"],
        help="Output format (json only for Surface A).",
    )
    evaluate_parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Include extra diagnostic fields in the artifact.",
    )

    # Surface B subcommand — deliberately namespaced under governance so it
    # is not mistaken for a live/runtime integration path.
    validate_parser = gov_subparsers.add_parser(
        "validate-manifest",
        help="Validate a Surface B manifest fixture locally",
    )
    validate_parser.add_argument(
        "--manifest",
        dest="manifest",
        required=True,
        help="Path to the manifest JSON fixture to validate.",
    )
    validate_parser.add_argument(
        "--schema",
        dest="schema",
        required=False,
        help="Path to a trusted schema JSON file for identity checks.",
    )
    validate_parser.add_argument(
        "--policy",
        dest="policy",
        required=False,
        help="Path to a trusted policy bundle JSON file for identity checks.",
    )
    validate_parser.add_argument(
        "--vocabulary",
        dest="vocabulary",
        required=False,
        help="Path to a trusted vocabulary JSON file for identity checks.",
    )
    validate_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Directory where the validation report will be written.",
    )
    validate_parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Include extra diagnostic fields in the report.",
    )

    return gov_parser


def governance_command(args: argparse.Namespace) -> int:
    """Entry point dispatched by ``cmd_governance`` in main.py.

    Wraps unexpected exceptions so they map to EXIT_RUNTIME_ERROR (30)
    rather than leaking as the default Python shell exit code 1.
    """
    try:
        sub = getattr(args, "governance_command", None)
        if sub == "evaluate":
            return cmd_evaluate(args)
        if sub == "validate-manifest":
            return cmd_validate_manifest(args)
        # No subcommand or unknown subcommand: print usage via argparse (which
        # already happened if parsing was triggered) and fail-closed.
        print("governance: expected subcommand (evaluate, validate-manifest)", file=sys.stderr)
        return EXIT_FAIL_CLOSED
    except Exception as exc:
        logger.exception("Unhandled exception in governance_command")
        print(
            f"governance: unexpected runtime failure: {exc}",
            file=sys.stderr,
        )
        return EXIT_RUNTIME_ERROR


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a local governance input fixture and write a decision artifact.

    Returns one of the documented exit codes.  Always writes a clear local
    decision artifact when returning 0 or 10; never returns 0/10 when the
    artifact cannot be written or is not authoritative.
    """
    # Guard: refuse unless explicitly in dry-run/report-only mode.
    if not getattr(args, "dry_run", False):
        print(
            "governance evaluate: --dry-run is required for Surface A. "
            "Refusing to proceed.",
            file=sys.stderr,
        )
        return EXIT_FAIL_CLOSED

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    # -- Try to load input ----------------------------------------------------
    try:
        raw_data = _load_input(input_path)
    except FileNotFoundError:
        _write_decision(
            output_dir,
            {
                "decision": "deny",
                "reason": "Missing required input artifact",
                "input_path": str(input_path),
                "collector": COLLECTOR_IDENTITY,
                "authorization": False,
            },
        )
        return EXIT_FAIL_CLOSED
    except json.JSONDecodeError as exc:
        _write_decision(
            output_dir,
            {
                "decision": "deny",
                "reason": f"Malformed input: {exc}",
                "input_path": str(input_path),
                "collector": COLLECTOR_IDENTITY,
                "authorization": False,
            },
        )
        return EXIT_FAIL_CLOSED

    # -- Validate structure (fail-closed) -------------------------------------
    structural_errors = _validate_structure(raw_data)
    if structural_errors:
        _write_decision(
            output_dir,
            {
                "decision": "deny",
                "reason": "Structurally incomplete input",
                "structural_errors": structural_errors,
                "input_path": str(input_path),
                "collector": COLLECTOR_IDENTITY,
                "authorization": False,
            },
        )
        return EXIT_FAIL_CLOSED

    # -- Validate known vocabulary (fail-closed) -------------------------------
    vocab_errors = _validate_vocabulary(raw_data)
    if vocab_errors:
        _write_decision(
            output_dir,
            {
                "decision": "deny_unknown",
                "reason": "Unknown/unmapped Kanban vocabulary detected",
                "vocabulary_errors": vocab_errors,
                "collector": COLLECTOR_IDENTITY,
                "authorization": False,
            },
        )
        return EXIT_FAIL_CLOSED

    # -- Policy evaluation (minimal Surface A stub) -----------------------------
    try:
        policy_result = _evaluate_policy(raw_data)
    except Exception as exc:  # pragma: no cover — safety net
        _write_decision(
            output_dir,
            {
                "decision": "deny",
                "reason": f"Unexpected policy evaluation failure: {exc}",
                "input_path": str(input_path),
                "collector": COLLECTOR_IDENTITY,
                "authorization": False,
            },
        )
        return EXIT_RUNTIME_ERROR

    artifact = {
        "decision": policy_result["decision"],
        "reason": policy_result["reason"],
        "input_path": str(input_path),
        "collector": COLLECTOR_IDENTITY,
        "authorization": False,  # explicit non-authorization for Surface A
        "policy_result": policy_result,
    }
    if getattr(args, "verbose", False):
        artifact["verbose"] = True
        artifact["raw_input_summary"] = _summarize_raw(raw_data)

    _write_decision(output_dir, artifact)

    if policy_result["decision"] == "deny":
        return EXIT_BLOCKED
    return EXIT_OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid Surface A input schema for this slice.
# Surface A only requires a known Kanban status; outcomes and other fields
# are optional.  Presence of a recognized status proves the fixture is
# structured enough to evaluate the minimal policy.
REQUIRED_STATUS_FIELD = "status"


def _load_input(path: Path) -> dict:
    """Read and parse a JSON input fixture.

    Raises json.JSONDecodeError for non-object JSON so the caller can
    map it to the fail-closed path uniformly.
    """
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise json.JSONDecodeError(
            f"Expected a JSON object, got {type(data).__name__}",
            doc=str(data)[:200],
            pos=0,
        )
    return data


def _validate_structure(data: dict) -> list[str]:
    """Validate the input object satisfies the minimal Surface A schema.

    Returns a list of error strings; empty list means structurally valid.
    """
    errors: list[str] = []
    # status is the minimal required field; without it the fixture is incomplete
    status = data.get(REQUIRED_STATUS_FIELD)
    if status is None:
        errors.append(f"Missing required field: {REQUIRED_STATUS_FIELD!r}")
    elif status not in VALID_STATUSES:
        # Unknown status is treated as a structural/schema failure (Surface A
        # relies on status presence) rather than a pure vocabulary mismatch.
        errors.append(f"Unknown or unsupported status: {status!r}")
    return errors


def _validate_vocabulary(data: dict) -> list[str]:
    """Return a list of error strings for unknown Kanban vocabulary.

    Empty list means all observed vocabulary is known.
    """
    errors: list[str] = []
    status = data.get("status")
    if status is not None and status not in VALID_STATUSES:
        errors.append(f"Unknown status: {status!r}")
    outcome = data.get("outcome")
    if outcome is not None and outcome not in VALID_OUTCOMES:
        errors.append(f"Unknown outcome: {outcome!r}")
    return errors


def _evaluate_policy(data: dict) -> dict:
    """Minimal policy evaluator for Surface A.

    Always returns non-authorization semantics.
"""
    status = data.get("status")
    if status == "blocked":
        return {"decision": "deny", "reason": "Task status is blocked"}
    # Any recognized status other than blocked is allowed by Surface A policy.
    # Structure validation is already complete, so status is guaranteed present
    # and known.
    return {"decision": "allow", "reason": "Surface A minimal policy: no blocking signal in input"}


def _write_decision(output_dir: Path, artifact: dict) -> Path:
    """Write the decision artifact to the output directory."""
    dest = output_dir / "decision.json"
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return dest

def _is_denylisted_path(path: Path) -> bool:
    """Return True if any path component matches denylisted patterns."""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    for part in resolved.parts:
        lowered = str(part).lower()
        for pat in _DENYLISTED_PATH_PATTERNS:
            if pat in lowered:
                return True
    return False


# ---------------------------------------------------------------------------
# Canonicalization hardening helpers
# ---------------------------------------------------------------------------


def _load_json_with_duplicate_guard(path: Path) -> tuple[dict | None, list[str]]:
    """Parse JSON with duplicate-key detection and size gate.

    Returns (data, errors).  If errors is non-empty, data may be None.
    """
    errors: list[str] = []
    try:
        st = path.stat()
    except FileNotFoundError:
        return (None, ["manifest_missing"])
    except Exception as exc:
        return (None, [f"manifest_unreadable:{exc}"])
    if st.st_size > _MAX_MANIFEST_SIZE_BYTES:
        return (None, ["manifest_exceeds_max_size"])

    try:
        with path.open("r", encoding="utf-8") as fh:
            # Duplicate-key guard: object_pairs_hook receives the raw list of
            # (key, value) pairs in document order.  We detect duplicates here.
            def _reject_duplicate_keys(pairs):
                seen = set()
                for key, _ in pairs:
                    if key in seen:
                        raise json.JSONDecodeError(
                            f"Duplicate key: {key!r}", doc=str(key), pos=0
                        )
                    seen.add(key)
                return dict(pairs)

            data = json.load(fh, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        if getattr(exc, "msg", "").startswith("Duplicate key"):
            return (None, ["manifest_duplicate_keys"])
        return (None, ["manifest_malformed"])
    except Exception as exc:
        return (None, [f"manifest_unreadable:{exc}"])

    if not isinstance(data, dict):
        return (None, ["manifest_not_an_object"])
    return (data, [])


def _json_depth(obj: Any, _seen: set[int] | None = None) -> int:
    """Return max nesting depth for dicts/lists.  Scalars = 0.

    Returns -1 if a circular reference is detected so callers can fail closed
    without letting RecursionError leak to the CLI surface.
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return -1
    if isinstance(obj, dict):
        if not obj:
            return 1
        _seen.add(obj_id)
        try:
            child_depths = [_json_depth(v, _seen) for v in obj.values()]
        finally:
            _seen.discard(obj_id)
        if any(d < 0 for d in child_depths):
            return -1
        return 1 + max(child_depths, default=0)
    if isinstance(obj, list):
        if not obj:
            return 1
        _seen.add(obj_id)
        try:
            child_depths = [_json_depth(v, _seen) for v in obj]
        finally:
            _seen.discard(obj_id)
        if any(d < 0 for d in child_depths):
            return -1
        return 1 + max(child_depths, default=0)
    return 0


def _validate_manifest_canonicalization(manifest: dict, out: list[str]) -> None:
    """Canonicalization hardening checks: depth, version, serializability."""
    depth = _json_depth(manifest)
    if depth < 0 or depth > _MAX_MANIFEST_DEPTH:
        out.append("manifest_exceeds_max_depth")
    version = manifest.get("manifest_version")
    if version not in _KNOWN_MANIFEST_VERSIONS:
        out.append(f"manifest_version_unsupported:{version}")
    # Guard: try to canonicalize the manifest (excluding bundle_id).
    # This catches circular references and non-JSON-serializable values
    # deterministically, mapping them to a canonicalization error rather
    # than letting RecursionError leak as EXIT_RUNTIME_ERROR.
    bundle_input = {k: v for k, v in manifest.items() if k != "bundle_id"}
    try:
        _canonical_json(bundle_input)
    except (TypeError, ValueError, RecursionError) as exc:
        out.append(f"manifest_canonicalization_invalid:{type(exc).__name__}")


def _canonical_json(obj: Any) -> str:
    """Serialize *obj* as canonical JSON (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_canonical_bundle_id(manifest: dict) -> str:
    """Compute canonical sha256 bundle digest excluding bundle_id itself."""
    bundle_identity_input = {k: v for k, v in manifest.items() if k != "bundle_id"}
    canonical_bytes = _canonical_json(bundle_identity_input).encode("utf-8")
    import hashlib

    return f"sha256:{hashlib.sha256(canonical_bytes).hexdigest()}"


# Denylisted path classes for referenced local files.
# Fail closed before reading/hash if any referenced path matches.
_DENYLISTED_PATH_PATTERNS = (
    ".env",
    ".env.",
    ".ssh",
    ".git",
    "kanban.db",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "secret",
    "token",
    "password",
    "credential",
    "api_key",
    "private_key",
)


# ---------------------------------------------------------------------------
# Surface B validator implementation
# ---------------------------------------------------------------------------

def _validate_manifest_fields(manifest: dict, out: list[str]) -> None:
    """Check required Surface B manifest fields are present."""
    missing = MANIFEST_REQUIRED_FIELDS - manifest.keys()
    if missing:
        out.append(f"missing_required_fields:{','.join(sorted(missing))}")


def _validate_manifest_schema_identity(
    manifest: dict, trusted_schema: dict | None, out: list[str]
) -> None:
    """Validate schema identity against an explicit trusted schema dict."""
    schema_id = manifest.get("schema_identity")
    if not isinstance(schema_id, dict):
        out.append("schema_identity_mismatch:schema_identity is not an object")
        return
    if trusted_schema is not None:
        if not isinstance(trusted_schema, dict):
            out.append("schema_identity_mismatch:trusted_schema_not_an_object")
            return
        for key in ("schema_name", "schema_version", "schema_hash"):
            if schema_id.get(key) != trusted_schema.get(key):
                out.append(f"schema_identity_mismatch:{key}")


def _validate_bundle_id(manifest: dict, out: list[str]) -> None:
    """Validate bundle_id format and recomputed canonical digest."""
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str):
        out.append("bundle_id_not_canonical:not_a_string")
        return
    if not bundle_id.startswith("sha256:"):
        out.append("bundle_id_not_canonical:missing_sha256_prefix")
        return
    # Recompute canonical digest and compare.
    computed = _compute_canonical_bundle_id(manifest)
    if bundle_id != computed:
        out.append("bundle_canonicalization_invalid:recomputed_digest_mismatch")


def _validate_board_identity(manifest: dict, out: list[str]) -> None:
    """Validate board_identity presence, shape, and non-empty values."""
    board = manifest.get("board_identity")
    if not isinstance(board, dict):
        out.append("namespace.identity_missing:board_identity not an object")
        return
    for key in ("board_slug", "tenant", "kanban_db_identity", "identity_source", "identity_source_hash"):
        if key not in board:
            out.append(f"namespace.identity_missing:{key}")
    board_slug = board.get("board_slug")
    if not isinstance(board_slug, str) or not board_slug.strip():
        out.append("namespace.identity_blank:board_slug")
    tenant = board.get("tenant")
    if tenant is not None and (not isinstance(tenant, str) or not tenant.strip()):
        out.append("namespace.identity_blank:tenant")
    kanban_db_identity = board.get("kanban_db_identity")
    if not isinstance(kanban_db_identity, dict) or not kanban_db_identity:
        out.append("namespace.identity_blank:kanban_db_identity")
    identity_source = board.get("identity_source")
    if not isinstance(identity_source, str) or not identity_source.strip():
        out.append("namespace.identity_blank:identity_source")
    identity_source_hash = board.get("identity_source_hash")
    if not isinstance(identity_source_hash, str) or not identity_source_hash.strip():
        out.append("namespace.identity_blank:identity_source_hash")


def _validate_workspace_identity(manifest: dict, out: list[str]) -> None:
    """Validate workspace_identity presence, shape, and non-dirty state."""
    ws = manifest.get("workspace_identity")
    if not isinstance(ws, dict):
        out.append("workspace.identity_missing:workspace_identity not an object")
        return
    kind = ws.get("workspace_kind")
    if kind not in {"scratch", "dir", "worktree"}:
        out.append(f"workspace.identity_invalid_kind:{kind}")
    path_class = ws.get("workspace_path_class")
    if not isinstance(path_class, str) or not path_class.strip():
        out.append("workspace.identity_blank:workspace_path_class")
    base_sha = ws.get("base_sha")
    if not isinstance(base_sha, str) or not base_sha.strip():
        out.append("workspace.identity_blank:base_sha")
    elif base_sha == "dirty":
        out.append("workspace.identity_dirty:base_sha")


def _validate_redaction_and_authority(manifest: dict, out: list[str]) -> None:
    """Validate redaction_state and authority_class are in allowed sets."""
    redaction = manifest.get("redaction_state")
    if redaction not in ALLOWED_REDACTION_STATES:
        out.append(f"redaction_state_invalid:{redaction}")
    authority = manifest.get("authority_class")
    if authority not in ALLOWED_AUTHORITY_CLASSES:
        out.append(f"authority_class_invalid:{authority}")
    # Surface B: authoritative_decision is not authorized yet.
    if authority == "authoritative_decision":
        out.append("authority_not_authorized_for_surface:authoritative_decision")


def _validate_artifact_refs(
    manifest: dict, out: list[str]
) -> list[dict]:
    """Validate explicit artifact_refs: denylisted paths, duplicate IDs, etc.

    Returns the list of refs that passed boundary checks for downstream
    TOCTOU/hash verification. Denylisted, traversal, or incomplete refs
    are *not* returned so they are never read or hashed.
    """
    refs = manifest.get("artifact_refs", [])
    if not isinstance(refs, list):
        out.append("artifact_refs_not_a_list")
        return []
    seen_ids: set[str] = set()
    seen_paths: dict[str, dict] = {}
    validated: list[dict] = []
    REQUIRED_FIELDS = ("artifact_id", "artifact_type", "path", "sha256", "size", "redaction_state", "authority_class")
    for ref in refs:
        if not isinstance(ref, dict):
            out.append("artifact_ref_invalid:not_an_object")
            continue
        # Require explicit file-reference fields
        missing = [f for f in REQUIRED_FIELDS if ref.get(f) is None or ref.get(f) == ""]
        if missing:
            out.append(f"artifact_ref_incomplete:{','.join(missing)}")
            continue
        # Type/format checks
        size = ref.get("size")
        if not isinstance(size, int) or size < 0:
            out.append(f"artifact_ref_invalid:size_not_int:{ref.get('artifact_id')}")
            continue
        sha256_val = ref.get("sha256", "")
        raw_hex = sha256_val[7:] if sha256_val.startswith("sha256:") else sha256_val
        if not (len(raw_hex) == 64 and all(c in "0123456789abcdefABCDEF" for c in raw_hex)):
            out.append(f"artifact_ref_invalid:sha256_format:{ref.get('artifact_id')}")
            continue
        redaction = ref.get("redaction_state")
        if redaction not in ALLOWED_REDACTION_STATES:
            out.append(f"artifact_ref_invalid:redaction_state:{redaction}")
            continue
        authority = ref.get("authority_class")
        if authority not in ALLOWED_AUTHORITY_CLASSES:
            out.append(f"artifact_ref_invalid:authority_class:{authority}")
            continue
        aid = ref.get("artifact_id")
        if aid is not None and aid in seen_ids:
            out.append(f"artifact_ref_duplicate_id:{aid}")
        if aid is not None:
            seen_ids.add(aid)
        path_str = ref.get("path")
        blocked = False
        if path_str:
            p = Path(path_str)
            if _is_denylisted_path(p):
                out.append(f"read_boundary_denylisted_path:{path_str}")
                blocked = True
            if ".." in p.parts:
                out.append(f"read_boundary_denylisted_path:traversal_in_path:{path_str}")
                blocked = True
            if not blocked:
                prev = seen_paths.get(str(p))
                if prev is not None:
                    if prev.get("sha256") != ref.get("sha256"):
                        out.append(f"artifact_ref_conflicting_hash:{path_str}")
                    if prev.get("authority_class") != ref.get("authority_class"):
                        out.append(f"artifact_ref_conflicting_authority:{path_str}")
                    if prev.get("redaction_state") != ref.get("redaction_state"):
                        out.append(f"artifact_ref_conflicting_redaction_or_export:{path_str}")
                seen_paths[str(p)] = ref
        if not blocked:
            validated.append(ref)
    return validated


def _toctou_check(path: Path, expected_size: int | None, out: list[str]) -> bool:
    """Basic TOCTOU: stat file before and after opening; must be regular file.

    Returns True if the file passed the check.
    """
    try:
        # Pre-stat
        pre_stat = path.stat()
        if not path.is_file():
            out.append(f"toctOU.unsafe_file_type:{path}")
            return False
        if path.is_symlink():
            out.append(f"toctOU.unsafe_file_type:symlink:{path}")
            return False
        with path.open("rb") as fh:
            # Post-stat from same handle where feasible (we re-stat path)
            post_stat = path.stat()
            if pre_stat.st_ino != post_stat.st_ino or pre_stat.st_size != post_stat.st_size:
                out.append(f"toctOU.file_identity_changed:{path}")
                return False
            # Read bytes for hash
            data = fh.read()
        if expected_size is not None and len(data) != expected_size:
            out.append(f"toctOU.file_changed_during_hash:size_mismatch:{path}")
            return False
        return True
    except FileNotFoundError:
        out.append(f"artifact_missing:{path}")
        return False
    except Exception as exc:
        out.append(f"toctOU.file_identity_changed:{path}:{exc}")
        return False


def _compute_sha256(path: Path) -> str | None:
    """Compute sha256 hex digest for a local file."""
    import hashlib

    try:
        with path.open("rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    """Validate a Surface B manifest fixture and write a diagnostic report.

    Local-only: reads explicit local files only. No live DB/API/network.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    errors: list[str] = []

    # No-op identity-check inputs are not accepted.
    if getattr(args, "policy", None):
        errors.append("policy_identity_check_not_implemented_surface_b")
    if getattr(args, "vocabulary", None):
        errors.append("vocabulary_identity_check_not_implemented_surface_b")

    # -- Load manifest --------------------------------------------------------
    manifest, load_errors = _load_json_with_duplicate_guard(manifest_path)
    if load_errors:
        errors.extend(load_errors)
        _write_validation_report(output_dir, False, errors, {})
        return SURFACE_B_EXIT_CODES["EXIT_SCHEMA_MISMATCH"]

    assert manifest is not None

    # -- Post-load canonicalization hardening -----------------------------------
    _validate_manifest_canonicalization(manifest, errors)
    # If canonicalization already failed, we still continue with field-level
    # checks so the report captures every relevant issue, but we will ultimately
    # return EXIT_SCHEMA_MISMATCH when any hardening error surfaces.

    # -- Field-level validations -----------------------------------------------
    _validate_manifest_fields(manifest, errors)
    trusted_schema: dict | None = None
    schema_arg = getattr(args, "schema", None)
    if schema_arg:
        sp = Path(schema_arg)
        if not sp.exists():
            errors.append("trusted_schema_missing")
        elif not sp.is_file():
            errors.append("trusted_schema_not_a_file")
        else:
            try:
                with sp.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    trusted_schema = raw
                else:
                    errors.append("trusted_schema_malformed:not_an_object")
            except Exception as exc:
                errors.append(f"trusted_schema_malformed:{exc}")
    _validate_manifest_schema_identity(manifest, trusted_schema, errors)
    _validate_bundle_id(manifest, errors)
    _validate_board_identity(manifest, errors)
    _validate_workspace_identity(manifest, errors)
    _validate_redaction_and_authority(manifest, errors)
    validated_refs = _validate_artifact_refs(manifest, errors)

    # -- TOCTOU + hash verification for explicit referenced files ---------------
    for ref in validated_refs:
        path_str = ref.get("path")
        if not path_str:
            continue
        ref_path = Path(path_str)
        if not _toctou_check(ref_path, ref.get("size"), errors):
            continue
        computed_hash = _compute_sha256(ref_path)
        expected_hash = ref.get("sha256")
        if computed_hash is None:
            errors.append(f"artifact_hash_unreadable:{path_str}")
        elif expected_hash and computed_hash != expected_hash:
            errors.append(f"stale_manifest.artifact_hash_mismatch:{path_str}")

    # -- Result ----------------------------------------------------------------
    # If canonicalization hardening produced any error, override the exit code
    # to EXIT_SCHEMA_MISMATCH (20) regardless of other validation outcomes.
    canon_errors = [e for e in errors if e.startswith(("manifest_exceeds_max", "manifest_duplicate_keys", "manifest_version_unsupported", "bundle_canonicalization_invalid"))]
    if canon_errors:
        _write_validation_report(output_dir, False, errors, manifest)
        return SURFACE_B_EXIT_CODES["EXIT_SCHEMA_MISMATCH"]
    valid = len(errors) == 0
    report = _write_validation_report(output_dir, valid, errors, manifest)
    return SURFACE_B_EXIT_CODES["EXIT_OK"] if valid else SURFACE_B_EXIT_CODES["EXIT_VALIDATION_FAILED"]


def _write_validation_report(
    output_dir: Path, valid: bool, errors: list[str], manifest: dict
) -> Path:
    """Write the local Surface B diagnostic validation report.

    This report is explicitly diagnostic-only and is NOT a consumer-safe
    manifest projection. Path redaction is applied but completeness is
    not guaranteed.
    """
    dest = output_dir / "validation_report.json"
    import re
    _POSIX_PATH = re.compile(r"/[^\s\"\:\[\]{}]+")
    _WIN_PATH  = re.compile(r"[A-Za-z]:[/\\][^\s\"\:\[\]{}]*")
    _URL       = re.compile(r"https?://[^\s\"\:\[\]{}]+")
    redacted_errors = []
    for e in errors:
        redacted = _POSIX_PATH.sub("<redacted_path>", e)
        redacted = _WIN_PATH.sub("<redacted_path>", redacted)
        redacted = _URL.sub("<redacted_url>", redacted)
        redacted_errors.append(redacted)
    report = {
        "valid": valid,
        "errors": redacted_errors,
        "authority_class": "diagnostic",
        "authorization": False,
        "report_type": "surface_b_diagnostic_validation",
        "consumer_safe": False,
        "non_authorizations": [
            "no_live_db_reads",
            "no_dashboard_consumption",
            "no_network",
            "no_mutation",
            "no_push_merge_pr",
            "no_deploy_restart_credentials",
        ],
    }
    # Bounded manifest identity snapshot — diagnostic only.
    if "manifest_version" in manifest:
        report["manifest_version"] = manifest["manifest_version"]
    if "bundle_id" in manifest:
        report["bundle_id"] = manifest["bundle_id"]
    if "board_identity" in manifest:
        board = manifest["board_identity"]
        report["manifest_identity_snapshot"] = {
            "board_slug": board.get("board_slug"),
            "tenant": board.get("tenant"),
            "identity_source": board.get("identity_source"),
            "identity_source_hash": board.get("identity_source_hash"),
        }
    with dest.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return dest
