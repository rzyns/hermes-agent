import argparse
import json
from types import SimpleNamespace

from hermes_cli import update_cmd
from hermes_cli.subcommands.update import build_update_parser


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda args: None)
    return parser


def test_update_maintenance_capabilities_are_side_effect_free(monkeypatch, capsys):
    monkeypatch.setattr(
        update_cmd,
        "_run_update_maintenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("capability detection must not run maintenance")
        ),
    )
    args = _parser().parse_args(["update-maintenance", "--capabilities"])

    args.func(args)

    assert json.loads(capsys.readouterr().out) == {
        "schema": 1,
        "command": "update-maintenance",
        "requires_fresh_process": True,
    }


def test_update_maintenance_runs_shared_pipeline_without_source_reconciliation(
    monkeypatch,
):
    calls = []
    output_state = {"sentinel": True}
    opts = SimpleNamespace(
        active_lazy_features=[],
        active_tool_dependencies=[],
        pre_update_version="old",
        gw_input_fn=None,
        assume_yes=True,
    )

    monkeypatch.setattr(
        update_cmd._m(),
        "_install_hangup_protection",
        lambda gateway_mode=False: output_state,
    )
    monkeypatch.setattr(
        update_cmd._m(),
        "_finalize_update_output",
        lambda value: calls.append(("finalize", value)),
    )
    monkeypatch.setattr(
        update_cmd._m(),
        "_finalize_update_receipt",
        lambda code, detail: calls.append(("receipt", code, detail)),
    )
    monkeypatch.setattr(
        update_cmd,
        "_resolve_update_options",
        lambda args, gateway_mode: opts,
    )
    monkeypatch.setattr(
        update_cmd, "_begin_update_receipt_and_plan", lambda args: "plan"
    )
    monkeypatch.setattr(
        update_cmd._m(), "_run_pre_update_backup", lambda args: "snapshot"
    )
    monkeypatch.setattr(update_cmd, "_record_update_step", lambda *args: None)
    monkeypatch.setattr(
        update_cmd._m(), "_pause_windows_gateways_for_update", lambda: []
    )
    monkeypatch.setattr(update_cmd._m(), "_is_windows", lambda: False)
    monkeypatch.setattr(update_cmd, "_desktop_app_present", lambda path: True)
    monkeypatch.setattr(update_cmd, "_base_git_cmd", lambda: ["git"])
    monkeypatch.setattr(update_cmd, "_current_branch_name", lambda git_cmd: "main")
    monkeypatch.setattr(
        update_cmd, "_capture_head_sha", lambda git_cmd, root: "abc123"
    )
    monkeypatch.setattr(
        update_cmd,
        "_apply_update_maintenance",
        lambda git_cmd, branch, pre_pull_sha, resolved, **kwargs: calls.append(
            (
                "maintenance",
                git_cmd,
                branch,
                pre_pull_sha,
                resolved,
                kwargs,
            )
        ),
    )
    args = _parser().parse_args(["update-maintenance", "--yes", "--gateway"])

    args.func(args)

    maintenance = calls[0]
    assert maintenance[:5] == ("maintenance", ["git"], "main", None, opts)
    assert maintenance[5]["source_updated"] is False
    assert maintenance[5]["expected_sha"] == "abc123"
    assert calls[1:] == [
        ("receipt", 0, "completed at command boundary"),
        ("finalize", output_state),
    ]
