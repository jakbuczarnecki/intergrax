# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P1 L2 gates — evidence schema, manifest, and negative contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import tests.qualification.dist_scale_x.catalog as dist_scale_x_catalog
from tests.qualification.dist_scale_x._constants import DIST_SCALE_X_P1_START_HEAD
from tests.qualification.dist_scale_x.catalog import validate_dist_scale_x_catalog_mappings
from tests.qualification.dist_scale_x.evidence_manifest import (
    DistScaleXEvidenceManifest,
    catalog_integrity_digest,
    load_and_validate_manifest_file,
    validate_manifest_mapping,
    write_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    CorrectnessInvariantViolation,
    DistScaleXEvidenceRecord,
    DistScaleXMetricsSnapshot,
    EvidenceDisposition,
    HistoricalEvidenceReconciliation,
    evidence_record_from_mapping,
    resolve_effective_disposition,
    validate_evidence_record,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _minimal_pass_record(**overrides: object) -> DistScaleXEvidenceRecord:
    base = DistScaleXEvidenceRecord(
        record_id="rec-p1-smoke",
        code_sha=DIST_SCALE_X_P1_START_HEAD,
        topology_description="metadata-only P1 validation",
        environment_definition="local deterministic pytest",
        workload_id="WL-1",
        dimension_ids=("A", "B"),
        duration_seconds=1.0,
        metrics=DistScaleXMetricsSnapshot(),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    return replace(base, **overrides) if overrides else base


def test_l2_valid_evidence_record_passes_validation() -> None:
    record = _minimal_pass_record()
    assert validate_evidence_record(record) == ()


def test_l2_missing_code_sha_fails_validation() -> None:
    record = _minimal_pass_record(code_sha="")
    violations = validate_evidence_record(record)
    assert any("code_sha" in v for v in violations)


def test_l2_pass_with_missing_mandatory_fields_cannot_stand() -> None:
    record = _minimal_pass_record(code_sha="", disposition=EvidenceDisposition.PASS)
    assert validate_evidence_record(record)
    assert resolve_effective_disposition(record) == EvidenceDisposition.FAIL


def test_l2_correctness_violation_overrides_pass_disposition() -> None:
    record = _minimal_pass_record(
        correctness_violations=(CorrectnessInvariantViolation.DUPLICATE_SIDE_EFFECT,),
    )
    assert resolve_effective_disposition(record) == EvidenceDisposition.FAIL


def test_l2_structural_violations_override_pass_disposition() -> None:
    record = _minimal_pass_record()
    assert (
        resolve_effective_disposition(record, structural_violations=("catalog stale",))
        == EvidenceDisposition.FAIL
    )


def test_l2_performance_thresholds_blocked_while_not_locked() -> None:
    record = _minimal_pass_record(performance_threshold_metric_ids=("M-WL1-THR",))
    violations = validate_evidence_record(record)
    assert any("NOT_LOCKED" in v for v in violations)


def test_l2_not_sufficient_historical_cannot_pass_alone() -> None:
    record = _minimal_pass_record(
        historical_reconciliation=HistoricalEvidenceReconciliation.NOT_SUFFICIENT,
    )
    violations = validate_evidence_record(record)
    assert any("NOT SUFFICIENT" in v for v in violations)


def test_l2_manifest_roundtrip_under_tmp_proof_layout() -> None:
    run_id = "p1-metadata-smoke"
    record = _minimal_pass_record()
    manifest = DistScaleXEvidenceManifest(
        schema_version=1,
        run_id=run_id,
        code_sha=DIST_SCALE_X_P1_START_HEAD,
        catalog_integrity_sha256=catalog_integrity_digest(),
        environment_definition="local deterministic pytest",
        topology_description="metadata-only P1 validation",
        qualification_phase="P1",
        records=(record,),
    )
    path = write_manifest(_REPO_ROOT, manifest)
    assert path.is_file()
    assert ".tmp/proof/dist-scale-x" in path.as_posix()
    violations = load_and_validate_manifest_file(path)
    assert violations == ()


def test_l2_catalog_digest_binds_material_catalog_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_digest = catalog_integrity_digest()
    assert validate_dist_scale_x_catalog_mappings() == ()

    first = dist_scale_x_catalog.DIST_SCALE_X_DIMENSION_CATALOG[0]
    mutated_catalog = (
        replace(first, title=f"{first.title} — material-field probe"),
        *dist_scale_x_catalog.DIST_SCALE_X_DIMENSION_CATALOG[1:],
    )
    monkeypatch.setattr(dist_scale_x_catalog, "DIST_SCALE_X_DIMENSION_CATALOG", mutated_catalog)
    assert validate_dist_scale_x_catalog_mappings() == ()
    assert catalog_integrity_digest() != baseline_digest


def test_l2_manifest_rejects_stale_catalog_digest() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "bad-digest",
        "code_sha": DIST_SCALE_X_P1_START_HEAD,
        "catalog_integrity_sha256": "0" * 64,
        "environment_definition": "local",
        "topology_description": "local",
        "qualification_phase": "P1",
        "records": [],
    }
    violations = validate_manifest_mapping(payload)
    assert any("catalog_integrity_sha256" in v for v in violations)


def test_l2_evidence_record_from_mapping_negative() -> None:
    mapping = {
        "record_id": "x",
        "code_sha": "",
        "topology_description": "",
        "environment_definition": "",
        "workload_id": "",
        "dimension_ids": [],
        "duration_seconds": 0,
        "metrics": {},
        "fault_description": "",
        "disposition": "PASS",
    }
    record = evidence_record_from_mapping(mapping)
    assert validate_evidence_record(record)


def test_l2_manifest_file_negative_missing_keys(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_manifest.json"
    bad_path.write_text(json.dumps({"run_id": "only"}), encoding="utf-8")
    violations = load_and_validate_manifest_file(bad_path)
    assert violations
