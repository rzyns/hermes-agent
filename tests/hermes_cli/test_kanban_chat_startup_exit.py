"""Kanban workers must fail closed on fatal chat startup configuration errors."""

import sys
from types import SimpleNamespace

import pytest

from hermes_cli import main as main_mod


def _chat_args():
    return SimpleNamespace(
        continue_last=None,
        resume=None,
        no_restore_cwd=False,
        worktree=False,
        yolo=False,
        ignore_user_config=False,
        ignore_rules=False,
        safe_mode=False,
        source=None,
        model=None,
        provider=None,
        toolsets=None,
        skills=None,
        verbose=None,
        quiet=True,
        query="work kanban task t_bad_config",
        image=None,
        checkpoints=False,
        pass_session_id=False,
        max_turns=None,
        compact=False,
    )


def test_root_cli_loader_uses_core_path_and_registers_canonical_module(
    monkeypatch,
):
    import importlib.util

    loaded = SimpleNamespace()
    calls = []

    def fake_main(**_kwargs):
        return None

    class Loader:
        def exec_module(self, module):
            assert sys.modules["cli"] is module
            module.main = fake_main
            calls.append("executed")

    spec = SimpleNamespace(name="cli", loader=Loader())
    monkeypatch.setitem(sys.modules, "cli", SimpleNamespace())

    def fake_spec_from_file_location(name, path):
        calls.append((name, path))
        return spec

    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", fake_spec_from_file_location
    )
    monkeypatch.setattr(importlib.util, "module_from_spec", lambda _spec: loaded)

    assert main_mod._load_root_cli_main() is fake_main
    assert calls[0] == (
        "cli",
        main_mod.Path(main_mod.__file__).resolve().parents[1] / "cli.py",
    )
    assert calls[1] == "executed"


@pytest.mark.parametrize(
    "startup_error",
    [
        ValueError("Unknown provider 'gpt-5.6-sol'"),
        ValueError("unparseable model config"),
    ],
)
def test_cmd_chat_kanban_startup_errors_use_infra_exit(
    monkeypatch, startup_error,
):
    def fail_startup(**_kwargs):
        raise startup_error

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_bad_config")
    monkeypatch.setattr(main_mod, "_resolve_use_tui", lambda _args: False)
    monkeypatch.setattr(main_mod, "_apply_safe_mode", lambda _args: None)
    monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_load_root_cli_main", lambda: fail_startup)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.cmd_chat(_chat_args())

    from hermes_cli.kanban_db import KANBAN_INFRA_EXIT_CODE

    assert exc_info.value.code == KANBAN_INFRA_EXIT_CODE
