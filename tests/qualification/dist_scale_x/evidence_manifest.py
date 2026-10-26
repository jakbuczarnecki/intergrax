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
from tests.qualification.dist_scale_x.catalog import validate_dist_scale_x_catalog_mappings
from tests.qualification.dist_scale_x.evidence_schema import (
    DistScaleXEvidenceRecord,
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


def catalog_integrity_digest() -> str:
    violations = validate_dist_scale_x_catalog_mappings()
    payload = json.dumps({"violations": violations}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def proof_run_directory(repo_root: Path, run_id: str) -> Path:
    return repo_root / DIST_SCALE_X_PROOF_ROOT_REL / run_id


def manifest_path_for_run(repo_root: Path, run_id: str) -> Path:
    return proof_run_directory(repo_root, run_id) / DIST_SCALE_X_MANIFEST_FILENAME


def validate_manifest_mapping(payload: Mapping[str, Any]) -> tuple[str, ...]:
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
    return validate_manifest_mapping(payload)
