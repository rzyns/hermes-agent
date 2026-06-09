"""CLI commands for the kanban-cross-deps plugin.

Wires ``hermes kanban-cross-deps <subcommand>``:
  add     — register a canonical cross-board edge
  remove  — remove an edge by id or composite key
  list    — list/filter edges (with --json)
  status  — explain blocking state for a child task

All commands support ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from plugins.kanban_cross_deps.diagnostics import CrossBoardDiagnostics
from plugins.kanban_cross_deps.models import VALID_EDGE_KINDS
from plugins.kanban_cross_deps.store import CrossBoardRegistry


def _json_out(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str))


def _edge_as_dict(edge) -> dict[str, Any]:
    return edge.to_dict()


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def register_cli(subparser: argparse.ArgumentParser) -> None:
    """Build the ``hermes kanban-cross-deps`` argparse tree."""
    subs = subparser.add_subparsers(dest="kcd_command")

    # add
    add_p = subs.add_parser(
        "add",
        help="Register a canonical cross-board edge",
    )
    add_p.add_argument("--parent-board", required=True, help="Parent board slug")
    add_p.add_argument("--parent-id", required=True, help="Parent task id")
    add_p.add_argument("--child-board", required=True, help="Child board slug")
    add_p.add_argument("--child-id", required=True, help="Child task id")
    add_p.add_argument(
        "--kind",
        required=True,
        choices=sorted(VALID_EDGE_KINDS),
        help="Edge kind",
    )
    add_p.add_argument(
        "--blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the edge blocks scheduler promotion (default: true)",
    )
    add_p.add_argument(
        "--required-statuses",
        default=None,
        help='JSON list of parent statuses that satisfy the edge, e.g. ["done","archived"]',
    )
    add_p.add_argument(
        "--source",
        default="canonical",
        help="Provenance source label (default: canonical)",
    )
    add_p.add_argument(
        "--created-by",
        default=None,
        help="Actor who created the edge",
    )
    add_p.add_argument(
        "--metadata",
        default=None,
        help="JSON object of arbitrary metadata",
    )
    add_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    # remove
    rm_p = subs.add_parser(
        "remove",
        aliases=["rm"],
        help="Remove a canonical edge by id or composite key",
    )
    rm_p.add_argument("--id", default=None, help="Edge uuid")
    rm_p.add_argument("--parent-board", default=None)
    rm_p.add_argument("--parent-id", default=None)
    rm_p.add_argument("--child-board", default=None)
    rm_p.add_argument("--child-id", default=None)
    rm_p.add_argument(
        "--kind",
        default=None,
        choices=sorted(VALID_EDGE_KINDS),
    )
    rm_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    # list
    ls_p = subs.add_parser(
        "list",
        aliases=["ls"],
        help="List/filter canonical edges",
    )
    ls_p.add_argument("--parent-board", default=None)
    ls_p.add_argument("--parent-id", default=None)
    ls_p.add_argument("--child-board", default=None)
    ls_p.add_argument("--child-id", default=None)
    ls_p.add_argument(
        "--kind",
        default=None,
        choices=sorted(VALID_EDGE_KINDS),
    )
    ls_p.add_argument(
        "--blocking",
        type=lambda s: s.lower() in {"1", "true", "yes"},
        default=None,
        help="Filter by blocking flag (true/false/1/0)",
    )
    ls_p.add_argument(
        "--source",
        default=None,
    )
    ls_p.add_argument(
        "--limit",
        type=int,
        default=500,
    )
    ls_p.add_argument(
        "--offset",
        type=int,
        default=0,
    )
    ls_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    ls_p.add_argument(
        "--count",
        action="store_true",
        help="Print count only",
    )

    # status
    st_p = subs.add_parser(
        "status",
        help="Explain blocking state for a child task",
    )
    st_p.add_argument("--child-board", required=True)
    st_p.add_argument("--child-id", required=True)
    st_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    # diagnostics
    diag_p = subs.add_parser(
        "diagnostics",
        aliases=["diag"],
        help="Run cross-board dependency diagnostics",
    )
    diag_p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    subparser.set_defaults(func=kanban_cross_deps_command)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def kanban_cross_deps_command(args: argparse.Namespace) -> int:
    sub = getattr(args, "kcd_command", None)
    if not sub:
        print("usage: hermes kanban-cross-deps {add,remove,list,status,diagnostics}")
        return 2

    if sub in {"add", "remove", "list", "ls", "status", "diagnostics", "diag"}:
        return _dispatch_registry_command(args)

    print(f"Unknown kanban-cross-deps subcommand: {sub}", file=sys.stderr)
    return 2


def _dispatch_registry_command(args: argparse.Namespace) -> int:
    reg = CrossBoardRegistry()

    sub = getattr(args, "kcd_command", None)

    if sub == "add":
        return _cmd_add(args, reg)
    if sub in {"remove", "rm"}:
        return _cmd_remove(args, reg)
    if sub in {"list", "ls"}:
        return _cmd_list(args, reg)
    if sub == "status":
        return _cmd_status(args, reg)
    if sub in {"diagnostics", "diag"}:
        return _cmd_diagnostics(args)

    return 2


def _cmd_add(args: argparse.Namespace, reg: CrossBoardRegistry) -> int:
    required_statuses = None
    if args.required_statuses:
        try:
            required_statuses = json.loads(args.required_statuses)
            if not isinstance(required_statuses, list):
                raise ValueError("required_statuses must be a JSON list")
        except Exception as exc:
            _err(args, f"Invalid --required-statuses: {exc}")
            return 2

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except Exception as exc:
            _err(args, f"Invalid --metadata: {exc}")
            return 2

    # Cycle guard: reject new blocking edges that would close a cycle
    if args.blocking:
        diag = CrossBoardDiagnostics(registry=reg)
        if diag.would_create_cycle(
            parent_board=args.parent_board,
            parent_id=args.parent_id,
            child_board=args.child_board,
            child_id=args.child_id,
            blocking=True,
        ):
            msg = (
                "Adding this edge would create a blocking cycle. "
                "Remove an existing edge, change to --no-blocking, or resolve the cycle first."
            )
            _err(args, msg)
            return 1

    try:
        edge = reg.add(
            parent_board=args.parent_board,
            parent_id=args.parent_id,
            child_board=args.child_board,
            child_id=args.child_id,
            kind=args.kind,
            blocking=args.blocking,
            required_parent_statuses=required_statuses,
            source=args.source,
            created_by=args.created_by,
            metadata=metadata,
        )
    except ValueError as exc:
        _err(args, str(exc))
        return 1

    if args.json:
        _json_out({"ok": True, "edge": _edge_as_dict(edge)})
    else:
        print(f"Added edge {edge.id}")
        print(f"  {edge.parent_board}/{edge.parent_id} --[{edge.kind}]{' (blocking)' if edge.blocking else ''}--> {edge.child_board}/{edge.child_id}")
    return 0


def _cmd_remove(args: argparse.Namespace, reg: CrossBoardRegistry) -> int:
    removed = False
    if args.id:
        removed = reg.remove(args.id)
    else:
        missing = []
        for attr in ("parent_board", "parent_id", "child_board", "child_id", "kind"):
            if getattr(args, attr) is None:
                missing.append(f"--{attr.replace('_', '-')}")
        if missing:
            _err(args, f"Remove requires --id OR all of {', '.join(missing)}")
            return 2
        removed = reg.remove_by_composite(
            parent_board=args.parent_board,
            parent_id=args.parent_id,
            child_board=args.child_board,
            child_id=args.child_id,
            kind=args.kind,
        )

    if args.json:
        _json_out({"ok": removed})
    else:
        print("Removed" if removed else "Not found")
    return 0 if removed else 1


def _cmd_list(args: argparse.Namespace, reg: CrossBoardRegistry) -> int:
    if args.count:
        count = reg.count(
            parent_board=args.parent_board,
            parent_id=args.parent_id,
            child_board=args.child_board,
            child_id=args.child_id,
            kind=args.kind,
            blocking=args.blocking,
            source=args.source,
        )
        if args.json:
            _json_out({"count": count})
        else:
            print(count)
        return 0

    edges = reg.list_edges(
        parent_board=args.parent_board,
        parent_id=args.parent_id,
        child_board=args.child_board,
        child_id=args.child_id,
        kind=args.kind,
        blocking=args.blocking,
        source=args.source,
        limit=args.limit,
        offset=args.offset,
    )
    if args.json:
        _json_out({"edges": [_edge_as_dict(e) for e in edges], "count": len(edges)})
    else:
        print(f"Edges ({len(edges)}):")
        for e in edges:
            block_flag = "[B]" if e.blocking else "[N]"
            print(f"  {e.id} {block_flag} {e.parent_board}/{e.parent_id} --{e.kind}--> {e.child_board}/{e.child_id} (source={e.source})")
    return 0


def _cmd_status(args: argparse.Namespace, reg: CrossBoardRegistry) -> int:
    edges = reg.list_edges(
        child_board=args.child_board,
        child_id=args.child_id,
    )

    blocking_edges = [e for e in edges if e.blocking]
    non_blocking_edges = [e for e in edges if not e.blocking]

    # Distinguish canonical from inferred (if any non-canonical edges appear)
    canonical = [e for e in edges if e.source == "canonical"]
    inferred = [e for e in edges if e.source != "canonical"]

    result = {
        "child_board": args.child_board,
        "child_id": args.child_id,
        "total_edges": len(edges),
        "blocking_edges": len(blocking_edges),
        "non_blocking_edges": len(non_blocking_edges),
        "canonical_edges": len(canonical),
        "inferred_edges": len(inferred),
        "edges": [_edge_as_dict(e) for e in edges],
    }

    if args.json:
        _json_out(result)
    else:
        print(f"Status for {args.child_board}/{args.child_id}")
        print(f"  Total edges: {len(edges)}")
        print(f"  Blocking: {len(blocking_edges)}  Non-blocking: {len(non_blocking_edges)}")
        print(f"  Canonical: {len(canonical)}  Inferred/other: {len(inferred)}")
        if edges:
            print("  Edges:")
            for e in edges:
                block_flag = "[B]" if e.blocking else "[N]"
                print(f"    {e.id} {block_flag} {e.parent_board}/{e.parent_id} --{e.kind}--> {e.child_board}/{e.child_id} (source={e.source})")
    return 0


def _err(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "json", False):
        _json_out({"ok": False, "error": message})
    else:
        print(message, file=sys.stderr)


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    diag = CrossBoardDiagnostics()
    report = diag.run()
    if args.json:
        _json_out(report)
    else:
        summary = report.get("summary", {})
        print("Cross-Board Dependency Diagnostics")
        print(f"  Blocking cycles: {summary.get('blocking_cycles', 0)}")
        print(f"  Informational cycles: {summary.get('informational_cycles', 0)}")
        print(f"  Dangling edges: {summary.get('dangling', 0)}")
        print(f"  Contradictions: {summary.get('contradictions', 0)}")
        print(f"  Provider failures: {summary.get('provider_failures', 0)}")
        if report.get("dangling"):
            print("  Dangling:")
            for d in report["dangling"]:
                print(f"    {d['edge_id']} → {d['missing_side']}: {d['reason']}")
        if report.get("contradictions"):
            print("  Contradictions:")
            for c in report["contradictions"]:
                print(f"    {c['edge_id']} {c['kind']}: {c['reason']}")
        if report.get("cycles", {}).get("blocking"):
            print("  Blocking cycles:")
            for cycle in report["cycles"]["blocking"]:
                path = " -> ".join(cycle.get("path", []))
                print(f"    {path}")
    return 0