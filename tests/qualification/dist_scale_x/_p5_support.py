# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P5 physical qualification helpers."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from tests.qualification.dist_scale_x._p2_support import LOCK_ENV_DEFINITION, code_sha, rabbit_broker_ready
from tests.qualification.dist_scale_x._p3_support import new_run_id
from tests.qualification.dist_scale_x._p4_support import (
    require_p4_docker_infra,
    require_p4_pg_infra,
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

P5_ENVELOPE_RECORD_ID = "DIST-SCALE-X-ENVELOPE-P5-v1"
P5_ENV_DEFINITION = (
    f"{LOCK_ENV_DEFINITION}; P5 envelope {P5_ENVELOPE_RECORD_ID}; "
    "ENVELOPE_ONLY; NOT_LOCKED; LKW; Rabbit M"
)
P5_RABBIT_CONTAINER = "intergrax-rabbitmq"
P5_BROKER_OUTAGE_SECONDS = 25


def require_p5_infra() -> None:
    from tests.qualification.dist_scale_x._p2_support import require_p2_infra

    require_p2_infra()
    require_p4_pg_infra()


def require_p5_docker() -> None:
    require_p4_docker_infra()


def p5_record(
    *,
    record_id: str,
    workload_id: str,
    dimension_ids: tuple[str, ...],
    duration_seconds: float,
    topology_description: str,
    fault_description: str,
    metrics: DistScaleXMetricsSnapshot | None = None,
) -> DistScaleXEvidenceRecord:
    return DistScaleXEvidenceRecord(
        record_id=record_id,
        code_sha=code_sha(),
        topology_description=topology_description,
        environment_definition=P5_ENV_DEFINITION,
        workload_id=workload_id,
        dimension_ids=dimension_ids,
        duration_seconds=duration_seconds,
        metrics=metrics or DistScaleXMetricsSnapshot(),
        fault_description=fault_description,
        disposition=EvidenceDisposition.PASS,
    )


def write_p5_manifest(
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
        environment_definition=P5_ENV_DEFINITION,
        topology_description=topology_description,
        qualification_phase="P5",
        records=records,
    )
    return write_manifest(_REPO_ROOT, manifest)


def docker_stop_rabbit(container: str | None = None) -> None:
    name = container or P5_RABBIT_CONTAINER
    subprocess.run(
        ["docker", "stop", name],
        check=False,
        capture_output=True,
        timeout=60,
    )


def docker_start_rabbit(container: str | None = None) -> None:
    name = container or P5_RABBIT_CONTAINER
    subprocess.run(
        ["docker", "start", name],
        check=False,
        capture_output=True,
        timeout=60,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if rabbit_broker_ready():
            return
        time.sleep(2)
    raise RuntimeError(f"RabbitMQ not ready after start: {name}")


def bounded_rabbit_outage(seconds: int | None = None) -> None:
    window = seconds if seconds is not None else P5_BROKER_OUTAGE_SECONDS
    try:
        docker_stop_rabbit()
        time.sleep(window)
    finally:
        docker_start_rabbit()


__all__ = [
    "P5_BROKER_OUTAGE_SECONDS",
    "P5_ENVELOPE_RECORD_ID",
    "P5_ENV_DEFINITION",
    "P5_RABBIT_CONTAINER",
    "bounded_rabbit_outage",
    "docker_start_rabbit",
    "docker_stop_rabbit",
    "new_run_id",
    "p5_record",
    "require_p5_docker",
    "require_p5_infra",
    "write_p5_manifest",
]
