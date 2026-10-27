# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P2 physical qualification helpers (infra, metrics, manifests)."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

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

P2_POSTGRES_DSN_ENV = "INTERGRAX_POSTGRESQL_DSN"
P2_DEFAULT_PG_DSN = "postgresql://intergrax:intergrax@localhost:5434/intergrax"
P2_RABBIT_HOST = "localhost"
P2_RABBIT_PORT = 5672
P2_RABBIT_USER = "intergrax"
P2_RABBIT_PASS = "intergrax"

LOCK_ENV_DEFINITION = (
    "local Windows; docker compose postgresql:16.6 + rabbitmq:3.13-management; "
    "DSN postgresql://intergrax:intergrax@localhost:5434/intergrax"
)
LOCK_TOPOLOGY_TEMPLATE = (
    "LKW-qual: DocumentStoreTaskWorker OS processes + PostgreSQL DocumentStore + RabbitMQ M; "
    "N={n_workers}; in_flight={in_flight}; duration_s={duration_s}"
)

POINT_DURATION_SECONDS = int(os.environ.get("DIST_SCALE_X_P2_POINT_DURATION_SECONDS", "180"))
WL2_WARMUP_SECONDS = int(os.environ.get("DIST_SCALE_X_P2_WL2_WARMUP_SECONDS", "60"))
OVERLOAD_STEP_SECONDS = int(os.environ.get("DIST_SCALE_X_P2_OVERLOAD_SECONDS", "60"))


def code_sha() -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    )
    return out.strip()


def resolve_pg_dsn() -> str:
    return os.environ.get(P2_POSTGRES_DSN_ENV, P2_DEFAULT_PG_DSN).strip() or P2_DEFAULT_PG_DSN


def pg_port_open(dsn: str | None = None) -> bool:
    """PostgreSQL readiness — protocol handshake (TCP-only check is insufficient on Docker Desktop)."""
    dsn = dsn or resolve_pg_dsn()
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def rabbit_port_open() -> bool:
    try:
        with socket.create_connection((P2_RABBIT_HOST, P2_RABBIT_PORT), timeout=2.0):
            return True
    except OSError:
        return False


def rabbit_broker_ready() -> bool:
    if not rabbit_port_open():
        return False
    try:
        import pika  # type: ignore[import-untyped]
    except ImportError:
        return rabbit_port_open()
    try:
        params = pika.ConnectionParameters(
            host=P2_RABBIT_HOST,
            port=P2_RABBIT_PORT,
            credentials=pika.PlainCredentials(P2_RABBIT_USER, P2_RABBIT_PASS),
            socket_timeout=3.0,
        )
        conn = pika.BlockingConnection(params)
        conn.close()
        return True
    except Exception:
        return False


def require_p2_infra() -> None:
    if not pg_port_open():
        pytest.skip("PostgreSQL not reachable for DIST-SCALE-X P2 qual")
    if not rabbit_broker_ready():
        pytest.skip("RabbitMQ broker not ready for DIST-SCALE-X P2 qual")


@dataclass(slots=True)
class LatencySample:
    latency_ms: float


def percentile_ms(samples: list[float], pct: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[min(idx, len(ordered) - 1)]


def run_timed_submissions(
    *,
    duration_s: float,
    submit: Callable[[], None],
    max_in_flight: int,
    in_flight: list[float],
) -> tuple[int, list[float]]:
    deadline = time.monotonic() + duration_s
    submitted = 0
    latencies: list[float] = []
    while time.monotonic() < deadline:
        while len(in_flight) >= max_in_flight:
            time.sleep(0.01)
        start = time.monotonic()
        submit()
        submitted += 1
        in_flight.append(start)
        while in_flight and time.monotonic() - in_flight[0] > 0.05:
            latencies.append((time.monotonic() - in_flight.pop(0)) * 1000.0)
    while in_flight:
        latencies.append((time.monotonic() - in_flight.pop(0)) * 1000.0)
    return submitted, latencies


def write_p2_manifest(
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
        environment_definition=LOCK_ENV_DEFINITION,
        topology_description=topology_description,
        qualification_phase="P2",
        records=records,
    )
    return write_manifest(_REPO_ROOT, manifest)


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
