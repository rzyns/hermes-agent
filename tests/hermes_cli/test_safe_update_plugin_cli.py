import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace


PLUGIN_CLI = Path('/home/openclaw/.hermes/plugins/hermes-update-guard/cli.py')


def _load_cli():
    spec = importlib.util.spec_from_file_location('hermes_update_guard_cli_under_test', PLUGIN_CLI)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_update_cli_invokes_maintenance_when_source_changed(monkeypatch):
    cli = _load_cli()
    calls = []

    class Result:
        ok = True
        dry_run = False
        source_changed = True
        actions = []
        reasons = []
        warnings = []
        def summary(self):
            return 'summary'
        def to_dict(self):
            return {}

    monkeypatch.setattr(cli, 'safe_update', lambda *a, **k: Result())
    monkeypatch.setattr(
        cli,
        '_run_update_maintenance',
        lambda repo, yes=False, gateway=False, **_: calls.append((repo, yes, gateway)) or subprocess.CompletedProcess([], 0),
    )

    cli.main(SimpleNamespace(repo='/tmp/repo', dry_run=False, push=False, json=False, no_maintenance=False, yes=True, gateway=True, allow_dirty=False, backup=False, no_backup=False))

    assert calls == [('/tmp/repo', True, True)]


def test_safe_update_cli_skips_maintenance_on_dry_run(monkeypatch):
    cli = _load_cli()

    class Result:
        ok = True
        dry_run = True
        source_changed = True
        actions = []
        reasons = []
        warnings = []
        def summary(self):
            return 'summary'
        def to_dict(self):
            return {}

    monkeypatch.setattr(cli, 'safe_update', lambda *a, **k: Result())
    monkeypatch.setattr(
        cli,
        '_run_update_maintenance',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('dry-run must not run maintenance')),
    )

    cli.main(SimpleNamespace(repo='/tmp/repo', dry_run=True, push=False, json=False, no_maintenance=False, yes=False, gateway=False, allow_dirty=False, backup=False, no_backup=False))


def test_safe_update_cli_captures_maintenance_output_for_json(monkeypatch, capsys):
    cli = _load_cli()
    captures = []

    class Result:
        ok = True
        dry_run = False
        source_changed = True
        actions = []
        reasons = []
        warnings = []
        def summary(self):
            return 'summary'
        def to_dict(self):
            return {'ok': self.ok, 'actions': self.actions, 'reasons': self.reasons}

    monkeypatch.setattr(cli, 'safe_update', lambda *a, **k: Result())

    def fake_maintenance(repo, yes=False, gateway=False, capture=False):
        captures.append(capture)
        return subprocess.CompletedProcess([], 0, stdout='maintenance chatter')

    monkeypatch.setattr(cli, '_run_update_maintenance', fake_maintenance)

    cli.main(SimpleNamespace(repo='/tmp/repo', dry_run=False, push=False, json=True, no_maintenance=False, yes=False, gateway=False, allow_dirty=False, backup=False, no_backup=False))

    assert captures == [True]
    out = capsys.readouterr().out
    assert 'maintenance chatter' not in out
    assert '"ok": true' in out
