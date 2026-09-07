import argparse
import json
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_maint
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


@pytest.mark.parametrize("backup_mode", ["quick", "full"])
@pytest.mark.parametrize("no_backup", [False, True])
def test_update_maintenance_runs_shared_pipeline_without_source_reconciliation(
    monkeypatch, tmp_path, capsys, backup_mode, no_backup,
):
    calls = []
    backups = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"updates:\n  pre_update_backup: {backup_mode}\n", encoding="utf-8"
    )
    output_state = {"sentinel": True}
    opts = SimpleNamespace(
        active_lazy_features=[],
        active_tool_dependencies=[],
        pre_update_version="old",
        gw_input_fn=None,
        assume_yes=True,
    )

    monkeypatch.setattr(
        update_cmd,
        "_invalidate_update_cache",
        lambda: calls.append(("invalidate",)),
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
        update_cmd_maint, "_run_quick_snapshots",
        lambda: backups.append("quick") or "snapshot",
    )
    monkeypatch.setattr(
        update_cmd_maint, "_run_full_backup", lambda: backups.append("full")
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
    argv = ["update-maintenance", "--yes", "--gateway"]
    if no_backup:
        argv.append("--no-backup")
    args = _parser().parse_args(argv)

    args.func(args)

    assert calls[0] == ("invalidate",)
    maintenance = calls[1]
    assert maintenance[:5] == ("maintenance", ["git"], "main", None, opts)
    assert maintenance[5]["source_updated"] is False
    assert maintenance[5]["expected_sha"] == "abc123"
    assert maintenance[5]["pre_update_snapshot_id"] == (
        None if no_backup else "snapshot"
    )
    assert backups == ([] if no_backup else (
        ["quick", "full"] if backup_mode == "full" else ["quick"]
    ))
    assert ("skipped (--no-backup)" in capsys.readouterr().out) is no_backup
    assert calls[2:] == [
        ("receipt", 0, "completed at command boundary"),
        ("finalize", output_state),
    ]
