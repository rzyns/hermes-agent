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
# CLI wiring (main.py calls these)
# ---------------------------------------------------------------------------


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
        # No subcommand or unknown subcommand: print usage via argparse (which
        # already happened if parsing was triggered) and fail-closed.
        print("governance: expected subcommand (evaluate)", file=sys.stderr)
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


def _summarize_raw(data: dict) -> dict:
    """Return a shallow summary of the raw input for verbose mode."""
    return {
        "keys": sorted(data.keys())[:20],  # cap verbosity
        "has_status": "status" in data,
        "has_outcome": "outcome" in data,
    }
