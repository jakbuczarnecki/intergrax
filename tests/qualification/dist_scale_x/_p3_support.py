# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 physical qualification helpers."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p2_support import (
    LOCK_ENV_DEFINITION as P2_LOCK_ENV,
    code_sha,
    percentile_ms,
    require_p2_infra,
)
from tests.qualification.dist_scale_x.evidence_manifest import (
    DistScaleXEvidenceManifest,
    catalog_integrity_digest,
    write_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    DistScaleXEvidenceRecord,
    DistScaleXMetricsSnapshot,
    EvidenceDisposition,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

P3_ENVELOPE_RECORD_ID = "DIST-SCALE-X-ENVELOPE-P3-v1"
P3_ENV_DEFINITION = (
    f"{P2_LOCK_ENV}; P3 envelope {P3_ENVELOPE_RECORD_ID}; "
    "local HTTP loopback; ENVELOPE_ONLY; NOT_LOCKED"
)

WL2_WARMUP_SECONDS = int(os.environ.get("DIST_SCALE_X_P3_WL2_WARMUP_SECONDS", "60"))
WL2_MEASURE_SECONDS = int(os.environ.get("DIST_SCALE_X_P3_WL2_MEASURE_SECONDS", "120"))
WL3_WARMUP_SECONDS = int(os.environ.get("DIST_SCALE_X_P3_WL3_WARMUP_SECONDS", "30"))
WL3_MEASURE_SECONDS = int(os.environ.get("DIST_SCALE_X_P3_WL3_MEASURE_SECONDS", "120"))
WL4_SCENARIO_SECONDS = float(os.environ.get("DIST_SCALE_X_P3_WL4_SCENARIO_SECONDS", "180"))
WL6_FAULT_SECONDS = float(os.environ.get("DIST_SCALE_X_P3_WL6_FAULT_SECONDS", "120"))
WL6_DRAIN_CAP_SECONDS = float(os.environ.get("DIST_SCALE_X_P3_WL6_DRAIN_CAP_SECONDS", "600"))

P3_WORKERS = int(os.environ.get("DIST_SCALE_X_P3_WORKERS", "4"))
P3_MAX_IN_FLIGHT = int(os.environ.get("DIST_SCALE_X_P3_MAX_IN_FLIGHT", "2"))


def require_p3_infra() -> None:
    require_p2_infra()


def require_p3_pg_infra() -> None:
    """WL-2/WL-3 DocumentStore physical paths — PostgreSQL only."""
    from tests.qualification.dist_scale_x._p2_support import pg_port_open

    if not pg_port_open():
        import pytest

        pytest.skip("PostgreSQL not reachable for DIST-SCALE-X P3 qual")


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def write_p3_manifest(
    *,
    run_id: str,
    records: tuple[DistScaleXEvidenceRecord, ...],
    topology_description: str,
) -> Path:
    manifest = DistScaleXEvidenceManifest(
        schema_version=1,
        run_id=run_id,
        code_sha=code_sha(),
        catalog_integrity_sha256=catalog_integrity_digest(),
        environment_definition=P3_ENV_DEFINITION,
        topology_description=topology_description,
        qualification_phase="P3",
        records=records,
    )
    return write_manifest(_REPO_ROOT, manifest)


def p3_record(
    *,
    run_id: str,
    suffix: str,
    workload_id: str,
    dimension_ids: tuple[str, ...],
    duration_seconds: float,
    topology_description: str,
    fault_description: str,
    metrics: DistScaleXMetricsSnapshot,
) -> DistScaleXEvidenceRecord:
    return DistScaleXEvidenceRecord(
        record_id=f"{run_id}-{suffix}",
        code_sha=code_sha(),
        topology_description=topology_description,
        environment_definition=P3_ENV_DEFINITION,
        workload_id=workload_id,
        dimension_ids=dimension_ids,
        duration_seconds=duration_seconds,
        metrics=metrics,
        fault_description=fault_description,
        disposition=EvidenceDisposition.PASS,
    )


__all__ = [
    "P3_ENVELOPE_RECORD_ID",
    "P3_ENV_DEFINITION",
    "P3_MAX_IN_FLIGHT",
    "P3_WORKERS",
    "WL2_MEASURE_SECONDS",
    "WL2_WARMUP_SECONDS",
    "WL3_MEASURE_SECONDS",
    "WL3_WARMUP_SECONDS",
    "WL4_SCENARIO_SECONDS",
    "WL6_DRAIN_CAP_SECONDS",
    "WL6_FAULT_SECONDS",
    "code_sha",
    "new_run_id",
    "percentile_ms",
    "p3_record",
    "require_p3_infra",
    "write_p3_manifest",
]
