# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P4 physical qualification helpers."""

from __future__ import annotations

import os
import subprocess
import time

from tests.qualification.dist_scale_x._p2_support import pg_port_open, resolve_pg_dsn
from tests.qualification.dist_scale_x._p3_support import (
    code_sha,
    new_run_id,
    write_p3_manifest,
)

P4_ENVELOPE_RECORD_ID = "DIST-SCALE-X-ENVELOPE-P4-v1"
P4_ENV_DEFINITION = (
    "PostgreSQL M; 90/10 write/read; ENVELOPE_ONLY; NOT_LOCKED; "
    f"envelope {P4_ENVELOPE_RECORD_ID}"
)

WL5_WARMUP_SECONDS = int(os.environ.get("DIST_SCALE_X_P4_WL5_WARMUP_SECONDS", "60"))
WL5_MEASURE_SECONDS = int(os.environ.get("DIST_SCALE_X_P4_WL5_MEASURE_SECONDS", "120"))
WL5_WRITERS = int(os.environ.get("DIST_SCALE_X_P4_WL5_WRITERS", "2"))
WL5_WRITE_RATE_CAP = float(os.environ.get("DIST_SCALE_X_P4_WL5_WRITE_RATE_CAP", "40"))
WL5_PAYLOAD_BYTES = int(os.environ.get("DIST_SCALE_X_P4_WL5_PAYLOAD_BYTES", "1024"))
P4_FAULT_WINDOW_SECONDS = int(os.environ.get("DIST_SCALE_X_P4_FAULT_WINDOW_SECONDS", "30"))
P4_PG_CONTAINER = os.environ.get("DIST_SCALE_X_P4_PG_CONTAINER", "intergrax-postgresql")


def require_p4_pg_infra() -> None:
    if not pg_port_open():
        import pytest

        pytest.fail("QUALIFICATION BLOCKED: PostgreSQL handshake failed for DIST-SCALE-X P4")


def require_p4_docker_infra() -> None:
    import pytest

    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        pytest.fail(f"QUALIFICATION BLOCKED: Docker required for F-STORAGE ({exc})")


def qual_pg_schema(run_id: str) -> str:
    safe = run_id.replace("-", "_")[:32]
    return f"dist_scale_x_p4_{safe}"


def prime_p4_qualification_stores(dsn: str, schema: str) -> None:
    """Initialize PG relational + document DDL once before multi-process WL-5 workers."""
    from intergrax.integrations.providers.document_store.postgresql.bundle import (
        create_postgresql_document_store,
    )
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.execution.decision_finalization_conformance import (
        conformance_artifact_payload_codec_registry,
    )
    from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
        PostgreSQLTaskCheckpointStore,
    )
    from intergrax.runtime.persistence.postgresql_opens import open_postgresql_decision_checkpoint_store

    ensure_qual_pg_schema(dsn, schema)
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )
    PostgreSQLTaskCheckpointStore(config=config)
    open_postgresql_decision_checkpoint_store(
        config=config,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    create_postgresql_document_store(tenant_schema=schema, dsn=dsn)


def ensure_qual_pg_schema(dsn: str, schema: str) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, connect_timeout=5) as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)),
                )
            conn.commit()
        except psycopg.errors.UniqueViolation:
            conn.rollback()


def write_p4_manifest(**kwargs):  # type: ignore[no-untyped-def]
    from tests.qualification.dist_scale_x.evidence_manifest import (
        DistScaleXEvidenceManifest,
        catalog_integrity_digest,
        write_manifest,
    )
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    manifest = DistScaleXEvidenceManifest(
        schema_version=1,
        run_id=kwargs["run_id"],
        code_sha=code_sha(),
        catalog_integrity_sha256=catalog_integrity_digest(),
        environment_definition=kwargs.get("environment_definition", P4_ENV_DEFINITION),
        topology_description=kwargs["topology_description"],
        qualification_phase="P4",
        records=kwargs["records"],
    )
    return write_manifest(repo, manifest)


def docker_pause_pg(container: str | None = None) -> None:
    name = container or P4_PG_CONTAINER
    subprocess.run(["docker", "pause", name], check=True, capture_output=True)


def docker_unpause_pg(container: str | None = None) -> None:
    name = container or P4_PG_CONTAINER
    subprocess.run(["docker", "unpause", name], check=True, capture_output=True)


def bounded_pg_outage(seconds: int | None = None) -> None:
    window = seconds if seconds is not None else P4_FAULT_WINDOW_SECONDS
    docker_pause_pg()
    time.sleep(window)
    docker_unpause_pg()


__all__ = [
    "P4_ENVELOPE_RECORD_ID",
    "P4_ENV_DEFINITION",
    "WL5_MEASURE_SECONDS",
    "WL5_PAYLOAD_BYTES",
    "WL5_WARMUP_SECONDS",
    "WL5_WRITE_RATE_CAP",
    "WL5_WRITERS",
    "P4_FAULT_WINDOW_SECONDS",
    "P4_PG_CONTAINER",
    "bounded_pg_outage",
    "code_sha",
    "docker_pause_pg",
    "docker_unpause_pg",
    "new_run_id",
    "ensure_qual_pg_schema",
    "prime_p4_qualification_stores",
    "qual_pg_schema",
    "require_p4_docker_infra",
    "require_p4_pg_infra",
    "resolve_pg_dsn",
    "write_p3_manifest",
    "write_p4_manifest",
]
