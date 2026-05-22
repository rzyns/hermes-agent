from types import SimpleNamespace

from hermes_cli import main as hermes_main


def test_update_maintenance_capabilities(capsys):
    hermes_main.cmd_update_maintenance(SimpleNamespace(capabilities=True))

    out = capsys.readouterr().out
    assert '"command": "update-maintenance"' in out
    assert '"requires_fresh_process": true' in out


def test_update_maintenance_runs_shared_helper_without_git(monkeypatch):
    calls = []

    monkeypatch.setattr(hermes_main, "_install_hangup_protection", lambda gateway_mode=False: {"sentinel": True})
    monkeypatch.setattr(hermes_main, "_finalize_update_output", lambda state: calls.append(("finalize", state)))

    def fake_maintenance(args, *, gateway_mode=False, source_updated=True):
        calls.append(("maintenance", gateway_mode, source_updated, getattr(args, "yes", None)))

    monkeypatch.setattr(hermes_main, "_run_update_maintenance", fake_maintenance)
    monkeypatch.setattr(
        hermes_main.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("update-maintenance must not call subprocess.run directly")),
    )

    hermes_main.cmd_update_maintenance(SimpleNamespace(capabilities=False, gateway=True, yes=True))

    assert ("maintenance", True, False, True) in calls
    assert ("finalize", {"sentinel": True}) in calls
