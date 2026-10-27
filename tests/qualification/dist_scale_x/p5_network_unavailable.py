# © Artur Czarnecki. All rights reserved.

"""P5 dimension I — connectivity / unavailable dependency (not consensus partition)."""

from __future__ import annotations

import threading
import time

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    docker_pause_pg,
    docker_unpause_pg,
    prime_p4_qualification_stores,
    qual_pg_schema,
    resolve_pg_dsn,
)
from tests.qualification.dist_scale_x._p5_support import (
    P5_BROKER_OUTAGE_SECONDS,
    bounded_rabbit_outage,
    new_run_id,
    p5_record,
    require_p5_docker,
    require_p5_infra,
    write_p5_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from tests.qualification.dist_scale_x.p4_persistence_op import execute_p4_persistence_op

pytestmark = [pytest.mark.qualification, pytest.mark.integration]


def test_p5_i_pg_pause_bounded_errors_recover() -> None:
    require_p5_infra()
    require_p5_docker()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("i-pg"))
    prime_p4_qualification_stores(dsn, schema)
    errors: list[str] = []
    successes = 0

    def _writer() -> None:
        nonlocal successes
        for i in range(30):
            result = execute_p4_persistence_op(
                dsn=dsn,
                schema=schema,
                tenant_id="tenant-a",
                op_index=i + 1,
                payload_pad="i" * 512,
            )
            if result.error:
                errors.append(result.error)
            else:
                successes += 1
            time.sleep(0.1)

    docker_pause_pg()
    try:
        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        thread.join(timeout=60)
    finally:
        docker_unpause_pg()

    run_id = new_run_id("p5-i-pg")
    topo = "I-02 PG pause; fail-closed writes; recovery after unpause"
    write_p5_manifest(
        run_id=run_id,
        topology_description=topo,
        records=(
            p5_record(
                record_id=f"{run_id}-pg",
                workload_id="WL-5",
                dimension_ids=("I",),
                duration_seconds=30.0,
                topology_description=topo,
                metrics=DistScaleXMetricsSnapshot(
                    extra={
                        "write_errors": float(len(errors)),
                        "write_success": float(successes),
                    },
                ),
                fault_description="docker pause intergrax-postgresql",
            ),
        ),
    )
    assert errors, "expected errors during PG pause"
    post = execute_p4_persistence_op(
        dsn=dsn,
        schema=schema,
        tenant_id="tenant-a",
        op_index=99,
        payload_pad="post" * 64,
    )
    assert not post.error, "writes must recover after unpause"


def test_p5_i_rabbit_outage_bounded_recovery() -> None:
    require_p5_infra()
    require_p5_docker()
    from tests.qualification.dist_scale_x._p2_support import rabbit_broker_ready

    assert rabbit_broker_ready()
    run_id = new_run_id("p5-i-rabbit")
    bounded_rabbit_outage(P5_BROKER_OUTAGE_SECONDS)
    assert rabbit_broker_ready()
    topo = "I-01 Rabbit stop/start; bounded outage"
    write_p5_manifest(
        run_id=run_id,
        topology_description=topo,
        records=(
            p5_record(
                record_id=f"{run_id}-rb",
                workload_id="WL-6",
                dimension_ids=("I",),
                duration_seconds=float(P5_BROKER_OUTAGE_SECONDS),
                topology_description=topo,
                fault_description=f"docker stop intergrax-rabbitmq {P5_BROKER_OUTAGE_SECONDS}s",
            ),
        ),
    )
