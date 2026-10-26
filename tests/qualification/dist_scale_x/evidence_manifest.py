# © Artur Czarnecki. All rights reserved.

"""Evidence artifact layout under `.tmp/proof/dist-scale-x/<run-id>/` (metadata-only validation in P1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tests.qualification.dist_scale_x._constants import (
    DIST_SCALE_X_MANIFEST_FILENAME,
    DIST_SCALE_X_MANIFEST_SCHEMA_VERSION,
    DIST_SCALE_X_PROOF_ROOT_REL,
)
import tests.qualification.dist_scale_x.catalog as dist_scale_x_catalog
from tests.qualification.dist_scale_x.catalog import (
    DIST_SCALE_X_MANIFEST_QUALIFICATION_PHASES,
    DistScaleBackendEntry,
    DistScaleDimensionEntry,
    DistScaleExitCriterionEntry,
    DistScaleHistoricalEvidenceEntry,
    DistScaleWorkloadEntry,
    validate_dist_scale_x_catalog_mappings,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    DistScaleXEvidenceRecord,
    EvidenceDisposition,
    evidence_record_from_mapping,
    validate_evidence_record,
)


@dataclass(frozen=True, slots=True)
class DistScaleXEvidenceManifest:
    schema_version: int
    run_id: str
    code_sha: str
    catalog_integrity_sha256: str
    environment_definition: str
    topology_description: str
    qualification_phase: str
    records: tuple[DistScaleXEvidenceRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "code_sha": self.code_sha,
            "catalog_integrity_sha256": self.catalog_integrity_sha256,
            "environment_definition": self.environment_definition,
            "topology_description": self.topology_description,
            "qualification_phase": self.qualification_phase,
            "records": [
                {
                    "record_id": r.record_id,
                    "code_sha": r.code_sha,
                    "topology_description": r.topology_description,
                    "environment_definition": r.environment_definition,
                    "workload_id": r.workload_id,
                    "dimension_ids": list(r.dimension_ids),
                    "duration_seconds": r.duration_seconds,
                    "metrics": {
                        "throughput_per_sec": r.metrics.throughput_per_sec,
                        "p95_latency_ms": r.metrics.p95_latency_ms,
                        "p99_latency_ms": r.metrics.p99_latency_ms,
                        "error_rate": r.metrics.error_rate,
                        "max_queue_depth": r.metrics.max_queue_depth,
                        "retry_amplification_factor": r.metrics.retry_amplification_factor,
                        "extra": dict(r.metrics.extra),
                    },
                    "fault_description": r.fault_description,
                    "disposition": r.disposition.value,
                    "container_service_versions": list(r.container_service_versions),
                    "correctness_violations": [v.value for v in r.correctness_violations],
                    "historical_reconciliation": (
                        r.historical_reconciliation.value if r.historical_reconciliation else None
                    ),
                    "performance_threshold_metric_ids": list(r.performance_threshold_metric_ids),
                    "mandatory_fields_complete": r.mandatory_fields_complete,
                }
                for r in self.records
            ],
        }


def _canonical_dimension_entry(entry: DistScaleDimensionEntry) -> dict[str, Any]:
    return {
        "dimension_id": entry.dimension_id,
        "title": entry.title,
        "workload_ids": list(entry.workload_ids),
        "exit_criterion_ids": list(entry.exit_criterion_ids),
        "implementation_batch": entry.implementation_batch,
        "p1_structural_gate_node_ids": list(entry.p1_structural_gate_node_ids),
    }


def _canonical_workload_entry(entry: DistScaleWorkloadEntry) -> dict[str, Any]:
    return {
        "workload_id": entry.workload_id,
        "title": entry.title,
        "dimension_ids": list(entry.dimension_ids),
        "exit_criterion_ids": list(entry.exit_criterion_ids),
        "mandatory_backend_ids": list(entry.mandatory_backend_ids),
        "implementation_batch": entry.implementation_batch,
        "future_qual_owner_glob": entry.future_qual_owner_glob,
    }


def _canonical_exit_criterion_entry(entry: DistScaleExitCriterionEntry) -> dict[str, Any]:
    return {
        "exit_id": entry.exit_id,
        "title": entry.title,
        "dimension_ids": list(entry.dimension_ids),
        "workload_ids": list(entry.workload_ids),
        "evidence_layer": entry.evidence_layer,
    }


def _canonical_backend_entry(entry: DistScaleBackendEntry) -> dict[str, Any]:
    return {
        "backend_id": entry.backend_id,
        "name": entry.name,
        "qualification_class": entry.qualification_class.value,
        "code_anchor": entry.code_anchor,
        "qualification_owner": entry.qualification_owner,
        "implementation_batch": entry.implementation_batch,
        "operator_decision_required": entry.operator_decision_required,
    }


def _canonical_historical_evidence_entry(entry: DistScaleHistoricalEvidenceEntry) -> dict[str, Any]:
    return {
        "stage_id": entry.stage_id,
        "accepted_sha": entry.accepted_sha,
        "reconciliation": entry.reconciliation,
        "scope_note": entry.scope_note,
    }


def canonical_dist_scale_x_catalog_document() -> dict[str, Any]:
    """Deterministic, materially complete catalog body for integrity binding."""
    mapping_violations = validate_dist_scale_x_catalog_mappings()
    return {
        "catalog_integrity_schema": "dist-scale-x-catalog-integrity-v1",
        "structural_validation": {"violations": list(mapping_violations)},
        "dimensions": [
            _canonical_dimension_entry(e) for e in dist_scale_x_catalog.DIST_SCALE_X_DIMENSION_CATALOG
        ],
        "workloads": [
            _canonical_workload_entry(e) for e in dist_scale_x_catalog.DIST_SCALE_X_WORKLOAD_CATALOG
        ],
        "exit_criteria": [
            _canonical_exit_criterion_entry(e) for e in dist_scale_x_catalog.DIST_SCALE_X_EXIT_CRITERION_CATALOG
        ],
        "backends": [_canonical_backend_entry(e) for e in dist_scale_x_catalog.DIST_SCALE_X_BACKEND_CATALOG],
        "historical_evidence": [
            _canonical_historical_evidence_entry(e)
            for e in dist_scale_x_catalog.DIST_SCALE_X_HISTORICAL_EVIDENCE_CATALOG
        ],
        "threshold_lock_state": dist_scale_x_catalog.DIST_SCALE_X_THRESHOLD_LOCK_STATE.value,
    }


def catalog_integrity_digest() -> str:
    payload = json.dumps(
        canonical_dist_scale_x_catalog_document(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def proof_run_directory(repo_root: Path, run_id: str) -> Path:
    return repo_root / DIST_SCALE_X_PROOF_ROOT_REL / run_id


def manifest_path_for_run(repo_root: Path, run_id: str) -> Path:
    return proof_run_directory(repo_root, run_id) / DIST_SCALE_X_MANIFEST_FILENAME


def expected_run_id_from_manifest_path(path: Path) -> str | None:
    """Extract `<run_id>` when path is under `.tmp/proof/dist-scale-x/<run_id>/`."""
    parts = path.resolve().parts
    for index, part in enumerate(parts):
        if part == "dist-scale-x" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def validate_manifest_mapping(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str | None = None,
) -> tuple[str, ...]:
    violations: list[str] = []
    required_keys = (
        "schema_version",
        "run_id",
        "code_sha",
        "catalog_integrity_sha256",
        "environment_definition",
        "topology_description",
        "qualification_phase",
        "records",
    )
    for key in required_keys:
        if key not in payload:
            violations.append(f"manifest missing key: {key}")

    if payload.get("schema_version") != DIST_SCALE_X_MANIFEST_SCHEMA_VERSION:
        violations.append("manifest schema_version mismatch")

    if not str(payload.get("run_id", "")).strip():
        violations.append("manifest run_id empty")
    if not str(payload.get("code_sha", "")).strip():
        violations.append("manifest code_sha empty")
    if not str(payload.get("environment_definition", "")).strip():
        violations.append("manifest environment_definition empty")
    if not str(payload.get("topology_description", "")).strip():
        violations.append("manifest topology_description empty")

    phase_raw = str(payload.get("qualification_phase", "")).strip()
    if not phase_raw:
        violations.append("manifest qualification_phase empty")
    elif phase_raw not in DIST_SCALE_X_MANIFEST_QUALIFICATION_PHASES:
        violations.append("manifest qualification_phase not in catalog batch definitions")

    manifest_run_id = str(payload.get("run_id", "")).strip()
    if expected_run_id is not None and manifest_run_id:
        if manifest_run_id != expected_run_id:
            violations.append(
                "manifest run_id does not match proof run directory",
            )

    manifest_code_sha = str(payload.get("code_sha", ""))
    manifest_environment = str(payload.get("environment_definition", ""))
    manifest_topology = str(payload.get("topology_description", ""))

    catalog_violations = validate_dist_scale_x_catalog_mappings()
    if catalog_violations:
        for cv in catalog_violations:
            violations.append(f"catalog structural violation: {cv}")
        return tuple(violations)

    expected_digest = catalog_integrity_digest()
    if str(payload.get("catalog_integrity_sha256", "")) != expected_digest:
        violations.append("manifest catalog_integrity_sha256 stale or incorrect")

    records_raw = payload.get("records")
    if not isinstance(records_raw, list):
        violations.append("manifest records must be a list")
        return tuple(violations)

    for index, item in enumerate(records_raw):
        if not isinstance(item, Mapping):
            violations.append(f"record[{index}] is not an object")
            continue
        record = evidence_record_from_mapping(item)
        record_violations = validate_evidence_record(record)
        for rv in record_violations:
            violations.append(f"record[{index}]: {rv}")

        if record.disposition in (EvidenceDisposition.PASS, EvidenceDisposition.FAIL):
            if record.code_sha != manifest_code_sha:
                violations.append(f"record[{index}] code_sha does not match manifest code_sha")
            if record.environment_definition != manifest_environment:
                violations.append(
                    f"record[{index}] environment_definition does not match manifest",
                )
            if record.topology_description != manifest_topology:
                violations.append(
                    f"record[{index}] topology_description does not match manifest",
                )

    return tuple(violations)


def write_manifest(repo_root: Path, manifest: DistScaleXEvidenceManifest) -> Path:
    path = manifest_path_for_run(repo_root, manifest.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest.to_dict(), sort_keys=True, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


def load_and_validate_manifest_file(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ("manifest root must be object",)
    expected_run_id = expected_run_id_from_manifest_path(path)
    return validate_manifest_mapping(payload, expected_run_id=expected_run_id)
