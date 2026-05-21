"""Expanded Surface B validator tests covering the negative case matrix.

Strict TDD — each test is a single behavior.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def hermes_main():
    """Path to repo root for subprocess invocation."""
    return Path(__file__).resolve().parents[2]


def _canonical_bundle_id(data: dict) -> str:
    """Compute correct canonical bundle ID for a manifest dict."""
    bundle_identity_input = {k: v for k, v in data.items() if k != "bundle_id"}
    canonical = json.dumps(bundle_identity_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _make_manifest(**overrides) -> dict:
    """Return a minimal valid Surface B manifest."""
    m = {
        "manifest_version": "hgk.surface_b.manifest.v1",
        "schema_identity": {
            "schema_name": "test-fixture-schema",
            "schema_version": "1.0.0",
            "schema_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "policy_bundle_version": "1.0.0",
            "policy_bundle_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        },
        "bundle_id": "sha256:PLACEHOLDER",
        "task_id": "t_test_fixture_001",
        "run_id": 1,
        "task_status": "done",
        "run_status": "completed",
        "run_outcome": "completed",
        "workspace_identity": {
            "workspace_kind": "scratch",
            "workspace_path_class": "tmp_fixture",
            "base_sha": "not_applicable",
        },
        "artifact_refs": [],
        "collector_identity": {"name": "test-collector", "version": "0.0.0"},
        "collection_started_at": "2026-05-20T00:00:00Z",
        "collection_finished_at": "2026-05-20T00:00:01Z",
        "source_vocabulary": {
            "table_name": "test-vocab",
            "table_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        },
        "redaction_state": "raw_only",
        "authority_class": "diagnostic",
        "non_authorizations": ["no_live_db_reads"],
        "board_identity": {
            "board_slug": "test-board",
            "tenant": None,
            "kanban_db_identity": {
                "db_path_class": "fixture_db",
                "db_identity_sentinel": "live_db_not_read_surface_b_design_only",
            },
            "identity_source": "manifest_fixture",
            "identity_source_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        },
    }
    m.update(overrides)
    m["bundle_id"] = _canonical_bundle_id(m)
    return m


def _write_manifest(tmp_path: Path, **overrides) -> Path:
    """Write a minimal valid manifest fixture to a temp file."""
    fixture = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(_make_manifest(**overrides), indent=2))
    return fixture


def _write_schema(tmp_path: Path) -> Path:
    fixture = tmp_path / "schema.json"
    fixture.write_text(
        json.dumps(
            {
                "schema_name": "test-fixture-schema",
                "schema_version": "1.0.0",
                "schema_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            },
            indent=2,
        )
    )
    return fixture


def _run_validate_manifest(tmp_path: Path, hermes_main: Path, manifest: Path, schema: Path | None = None) -> subprocess.CompletedProcess:
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable, "-m", "hermes_cli.main",
        "governance", "validate-manifest",
        "--manifest", str(manifest),
        "--output-dir", str(out_dir),
    ]
    if schema is not None:
        cmd += ["--schema", str(schema)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(hermes_main))


# ---------------------------------------------------------------------------
# Requirement 4: schema expectations (required fields)
# ---------------------------------------------------------------------------
class TestSurfaceBRequiredFields:
    def test_missing_manifest_version_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        del m["manifest_version"]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        # After canonicalization hardening, missing manifest_version also
        # triggers manifest_version_unsupported → EXIT_SCHEMA_MISMATCH (20).
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("missing_required_fields" in e for e in report["errors"])

    def test_missing_board_identity_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        del m["board_identity"]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("missing_required_fields" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Requirement 5: bundle_id canonicalization
# ---------------------------------------------------------------------------
class TestSurfaceBBundleId:
    def test_random_uuid_bundle_id_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["bundle_id"] = "uuid:123e4567-e89b-12d3-a456-426614174000"
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("bundle_id_not_canonical" in e for e in report["errors"])

    def test_wrong_canonical_bundle_id_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["bundle_id"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        # After canonicalization hardening, recomputed_digest_mismatch
        # maps to EXIT_SCHEMA_MISMATCH (20).
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("bundle_canonicalization_invalid" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Requirement 6: board/tenant identity
# ---------------------------------------------------------------------------
class TestSurfaceBBoardIdentity:
    def test_missing_board_slug_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["board_identity"] = {
            "tenant": None,
            "kanban_db_identity": {},
            "identity_source": "fixture",
            "identity_source_hash": "sha256:0",
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_missing" in e for e in report["errors"])

    def test_tenant_mismatch_not_checked_in_local_validator(self, tmp_path, hermes_main):
        """Local validator does not have an expected tenant to compare; it only checks presence."""
        m = _make_manifest()
        m["board_identity"]["tenant"] = "some_tenant"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Requirement 7: redaction and authority class
# ---------------------------------------------------------------------------
class TestSurfaceBRedactionAuthority:
    def test_invalid_redaction_state_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["redaction_state"] = "maybe"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("redaction_state_invalid" in e for e in report["errors"])

    def test_authoritative_decision_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["authority_class"] = "authoritative_decision"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("authority_not_authorized_for_surface" in e for e in report["errors"])

    def test_invalid_authority_class_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["authority_class"] = "wizard"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("authority_class_invalid" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Requirement 8: schema identity against trusted local schema
# ---------------------------------------------------------------------------
class TestSurfaceBSchemaIdentity:
    def test_schema_mismatch_fails(self, tmp_path, hermes_main):
        schema = _write_schema(tmp_path)
        m = _make_manifest()
        m["schema_identity"]["schema_hash"] = "sha256:FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest, schema)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("schema_identity_mismatch" in e for e in report["errors"])

    def test_schema_match_passes(self, tmp_path, hermes_main):
        schema = _write_schema(tmp_path)
        manifest = _write_manifest(tmp_path)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest, schema)
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Requirement 9: denylisted paths fail closed
# ---------------------------------------------------------------------------
class TestSurfaceBDenylistedPaths:
    def test_denylisted_env_path_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "secret.env"),
                "sha256": "0" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("read_boundary_denylisted_path" in e for e in report["errors"])

    def test_traversal_in_path_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / ".." / "etc" / "passwd"),
                "sha256": "0" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("traversal_in_path" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Requirement 10: artifact ref conflicts
# ---------------------------------------------------------------------------
class TestSurfaceBArtifactRefConflicts:
    def test_duplicate_artifact_id_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file1.txt"),
                "sha256": "0" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            },
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file2.txt"),
                "sha256": "1" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            },
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_ref_duplicate_id" in e for e in report["errors"])

    def test_conflicting_hash_same_path_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file.txt"),
                "sha256": "0" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            },
            {
                "artifact_id": "a2",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file.txt"),
                "sha256": "1" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            },
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_ref_conflicting_hash" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Requirement B: canonicalization hardening (Option B)
# ---------------------------------------------------------------------------
class TestSurfaceBCanonicalization:
    def test_oversized_manifest_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        # Pad a string field to push manifest JSON over 1 MB
        m["padding"] = "x" * (2 * 1024 * 1024)
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_exceeds_max_size" in e for e in report["errors"])

    def test_deeply_nested_manifest_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        # Build a deeply nested dict (depth 64) under a new key
        payload = "leaf"
        for _ in range(64):
            payload = {"layer": payload}
        m["deep_payload"] = payload
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_exceeds_max_depth" in e for e in report["errors"])

    def test_duplicate_keys_manifest_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        # Inject a duplicate key by hand-crafting valid JSON text with two "manifest_version" keys
        raw = json.dumps(m, indent=2, ensure_ascii=False)
        # Insert a second *valid* manifest_version key before the first one ends
        raw_with_dup = raw.replace(
            '"manifest_version": "hgk.surface_b.manifest.v1"',
            '"manifest_version": "dup.v1",\n  "manifest_version": "hgk.surface_b.manifest.v1"',
            1,
        )
        manifest = tmp_path / "manifest.json"
        manifest.write_text(raw_with_dup)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_duplicate_keys" in e for e in report["errors"])

    def test_circular_reference_manifest_fails(self):
        """Direct unit test: circular dict triggers canonicalization guard.
        """
        from hermes_cli.governance import _validate_manifest_canonicalization
        m = {"manifest_version": "hgk.surface_b.manifest.v1", "task_id": "t1"}
        m["self"] = m
        errors: list[str] = []
        _validate_manifest_canonicalization(m, errors)
        assert any("manifest_canonicalization_invalid" in e for e in errors)

    def test_unsupported_manifest_version_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["manifest_version"] = "unsupported.version"
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_version_unsupported" in e for e in report["errors"])

    def test_digest_is_deterministic_across_key_order(self):
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        import hermes_cli.governance as gov
        assert gov._canonical_json(a) == gov._canonical_json(b)

    def test_non_canonical_input_triggers_mismatch(self, tmp_path, hermes_main):
        m = _make_manifest()
        # Write manifest with keys deliberately unsorted (non-canonical representation)
        # The validator should still recompute bundle_id correctly because json.load parses into dict,
        # but we can assert that _canonical_json serializes sorted.
        canonical = json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        unsorted = json.dumps(m, sort_keys=False, separators=(",", ":"), ensure_ascii=False)
        assert canonical != unsorted
        manifest = tmp_path / "manifest.json"
        manifest.write_text(unsorted)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        # Should still pass because json.load normalizes key order into dict memory layout
        assert result.returncode == 0
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert report["valid"] is True


class TestSurfaceBArtifactHashVerification:
    def test_missing_referenced_file_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "missing.txt"),
                "sha256": "0" * 64,
                "size": 0,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_missing" in e for e in report["errors"])

    def test_stale_hash_mismatch_fails(self, tmp_path, hermes_main):
        f = tmp_path / "test.txt"
        f.write_text("actual content")
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(f),
                "sha256": "0" * 64,
                "size": len(b"actual content"),
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("stale_manifest.artifact_hash_mismatch" in e for e in report["errors"])

    def test_matching_hash_passes(self, tmp_path, hermes_main):
        f = tmp_path / "test.txt"
        content = b"actual content"
        f.write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(f),
                "sha256": expected_hash,
                "size": len(content),
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 0
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert report["valid"] is True


# ---------------------------------------------------------------------------
# Requirement 12: consumer-safe report (raw paths not exposed in report)
# ---------------------------------------------------------------------------
class TestSurfaceBConsumerSafeReport:
    def test_report_no_raw_paths(self, tmp_path, hermes_main):
        f = tmp_path / "test.txt"
        f.write_text("content")
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(f),
                "sha256": "0" * 64,
                "size": 7,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        # Ensure raw artifact path is not directly exposed in report
        raw_report = (tmp_path / "out" / "validation_report.json").read_text()
        assert str(f) not in raw_report

    def test_report_is_diagnostic(self, tmp_path, hermes_main):
        manifest = _write_manifest(tmp_path)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 0
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert report["authority_class"] == "diagnostic"
        assert report["authorization"] is False


# ---------------------------------------------------------------------------
# Requirement 13: manifest file error handling
# ---------------------------------------------------------------------------
class TestSurfaceBManifestErrors:
    def test_missing_manifest_file_returns_schema_mismatch(self, tmp_path, hermes_main):
        missing = tmp_path / "no_manifest.json"
        result = _run_validate_manifest(tmp_path, hermes_main, missing)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert report["valid"] is False

    def test_malformed_manifest_returns_schema_mismatch(self, tmp_path, hermes_main):
        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        result = _run_validate_manifest(tmp_path, hermes_main, bad)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_malformed" in e or "manifest_malformed" in str(report["errors"]) for e in report["errors"])

    def test_non_object_manifest_returns_schema_mismatch(self, tmp_path, hermes_main):
        arr = tmp_path / "arr.json"
        arr.write_text("[]")
        result = _run_validate_manifest(tmp_path, hermes_main, arr)
        assert result.returncode == 20
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("manifest_not_an_object" in e for e in report["errors"])


# ---------------------------------------------------------------------------
# Regression tests for independent review repairs (t_97f80920 -> t_8bca1def)
# ---------------------------------------------------------------------------
class TestSurfaceBRepairDenylistBeforeRead:
    def test_denylisted_existing_file_with_wrong_hash_no_stale_error(self, tmp_path, hermes_main):
        """B1: an existing denylisted file must not be read/hashed after denylist detection.

        Even with a deliberately wrong hash and size, the only error for that path
        must be the denylist boundary error — no stale_manifest hash mismatch.
        """
        f = tmp_path / ".env"
        f.write_text("sensitive=secret")
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(f),
                "sha256": "0" * 64,
                "size": 999,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("read_boundary_denylisted_path" in e for e in report["errors"])
        assert not any("stale_manifest.artifact_hash_mismatch" in e for e in report["errors"])
        assert not any("artifact_hash_unreadable" in e for e in report["errors"])
        assert not any("toctOU" in e for e in report["errors"])


class TestSurfaceBRepairSchemaFailClosed:
    def test_missing_trusted_schema_file_fails_closed(self, tmp_path, hermes_main):
        """C1: --schema supplied but missing must fail closed with trusted_schema_missing."""
        manifest = _write_manifest(tmp_path)
        missing_schema = tmp_path / "no_schema.json"
        result = _run_validate_manifest(tmp_path, hermes_main, manifest, schema=missing_schema)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("trusted_schema_missing" in e for e in report["errors"])

    def test_trusted_schema_not_a_file_fails_closed(self, tmp_path, hermes_main):
        manifest = _write_manifest(tmp_path)
        bad = tmp_path / "not_a_file"
        bad.mkdir()
        result = _run_validate_manifest(tmp_path, hermes_main, manifest, schema=bad)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("trusted_schema_not_a_file" in e for e in report["errors"])

    def test_trusted_schema_malformed_object_fails_closed(self, tmp_path, hermes_main):
        manifest = _write_manifest(tmp_path)
        bad = tmp_path / "bad_schema.json"
        bad.write_text("\"not an object\"")
        result = _run_validate_manifest(tmp_path, hermes_main, manifest, schema=bad)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("trusted_schema_malformed" in e for e in report["errors"])


class TestSurfaceBRepairNoOpPolicyVocabulary:
    def test_policy_arg_rejected(self, tmp_path, hermes_main):
        """C2: --policy is accepted by argparse but must produce an error."""
        manifest = _write_manifest(tmp_path)
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable, "-m", "hermes_cli.main",
            "governance", "validate-manifest",
            "--manifest", str(manifest),
            "--output-dir", str(out_dir),
            "--policy", str(tmp_path / "policy.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hermes_main))
        assert result.returncode == 10
        report = json.loads((out_dir / "validation_report.json").read_text())
        assert any("policy_identity_check_not_implemented_surface_b" in e for e in report["errors"])

    def test_vocabulary_arg_rejected(self, tmp_path, hermes_main):
        """C2: --vocabulary is accepted by argparse but must produce an error."""
        manifest = _write_manifest(tmp_path)
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable, "-m", "hermes_cli.main",
            "governance", "validate-manifest",
            "--manifest", str(manifest),
            "--output-dir", str(out_dir),
            "--vocabulary", str(tmp_path / "vocab.json"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hermes_main))
        assert result.returncode == 10
        report = json.loads((out_dir / "validation_report.json").read_text())
        assert any("vocabulary_identity_check_not_implemented_surface_b" in e for e in report["errors"])


class TestSurfaceBRepairArtifactRefExplicitFields:
    def test_artifact_ref_missing_required_fields_fails(self, tmp_path, hermes_main):
        """C3: artifact refs without explicit path/sha256/size must be rejected."""
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file.txt"),
                # deliberately omit sha256 and size
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_ref_incomplete" in e for e in report["errors"])

    def test_artifact_ref_bad_sha256_format_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file.txt"),
                "sha256": "not_a_hash",
                "size": 0,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_ref_invalid:sha256_format" in e for e in report["errors"])

    def test_artifact_ref_negative_size_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["artifact_refs"] = [
            {
                "artifact_id": "a1",
                "artifact_type": "raw_snapshot",
                "authority_class": "diagnostic",
                "path": str(tmp_path / "file.txt"),
                "sha256": "0" * 64,
                "size": -1,
                "redaction_state": "raw_only",
            }
        ]
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("artifact_ref_invalid:size_not_int" in e for e in report["errors"])


class TestSurfaceBRepairIdentityStrength:
    def test_blank_board_slug_fails(self, tmp_path, hermes_main):
        """C4: blank board_slug must be invalid."""
        m = _make_manifest()
        m["board_identity"] = {
            "board_slug": "   ",
            "tenant": None,
            "kanban_db_identity": {"k": "v"},
            "identity_source": "fixture",
            "identity_source_hash": "sha256:" + "0" * 64,
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_blank:board_slug" in e for e in report["errors"])

    def test_empty_tenant_string_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["board_identity"] = {
            "board_slug": "valid-board",
            "tenant": "",
            "kanban_db_identity": {"k": "v"},
            "identity_source": "fixture",
            "identity_source_hash": "sha256:" + "0" * 64,
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_blank:tenant" in e for e in report["errors"])

    def test_empty_kanban_db_identity_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["board_identity"] = {
            "board_slug": "valid-board",
            "tenant": None,
            "kanban_db_identity": {},
            "identity_source": "fixture",
            "identity_source_hash": "sha256:" + "0" * 64,
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_blank:kanban_db_identity" in e for e in report["errors"])

    def test_blank_identity_source_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["board_identity"] = {
            "board_slug": "valid-board",
            "tenant": None,
            "kanban_db_identity": {"k": "v"},
            "identity_source": "",
            "identity_source_hash": "sha256:" + "0" * 64,
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_blank:identity_source" in e for e in report["errors"])

    def test_blank_identity_source_hash_fails(self, tmp_path, hermes_main):
        m = _make_manifest()
        m["board_identity"] = {
            "board_slug": "valid-board",
            "tenant": None,
            "kanban_db_identity": {"k": "v"},
            "identity_source": "fixture",
            "identity_source_hash": "",
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("namespace.identity_blank:identity_source_hash" in e for e in report["errors"])

    def test_dirty_workspace_identity_fails(self, tmp_path, hermes_main):
        """C5: dirty/unknown workspace identity must be invalid."""
        m = _make_manifest()
        m["workspace_identity"] = {
            "workspace_kind": "unknown",
            "workspace_path_class": "",
            "base_sha": "dirty",
        }
        m["bundle_id"] = _canonical_bundle_id(m)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps(m, indent=2))
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 10
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert any("workspace.identity_invalid_kind" in e for e in report["errors"])
        assert any("workspace.identity_blank:workspace_path_class" in e for e in report["errors"])
        assert any("workspace.identity_dirty:base_sha" in e for e in report["errors"])


class TestSurfaceBRepairReportHardening:
    def test_report_has_diagnostic_type_and_not_consumer_safe(self, tmp_path, hermes_main):
        """C6: report must declare itself diagnostic and not consumer-safe."""
        manifest = _write_manifest(tmp_path)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 0
        report = json.loads((tmp_path / "out" / "validation_report.json").read_text())
        assert report.get("report_type") == "surface_b_diagnostic_validation"
        assert report.get("consumer_safe") is False
        assert "board_slug" not in report  # not at top level
        assert "tenant" not in report      # not at top level
        assert "manifest_identity_snapshot" in report

    def test_report_top_level_no_raw_board_or_tenant(self, tmp_path, hermes_main):
        """C6: raw board_slug and tenant must not appear at report top level."""
        manifest = _write_manifest(tmp_path)
        result = _run_validate_manifest(tmp_path, hermes_main, manifest)
        assert result.returncode == 0
        raw = (tmp_path / "out" / "validation_report.json").read_text()
        report = json.loads(raw)
        assert "board_slug" not in report
        assert "tenant" not in report
        snapshot = report.get("manifest_identity_snapshot", {})
        assert snapshot.get("board_slug") == "test-board"
        assert snapshot.get("tenant") is None
