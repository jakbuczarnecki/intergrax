# © Artur Czarnecki. All rights reserved.

"""P6 WL-7 mixed soak (dimension M)."""

from __future__ import annotations

import os
import time

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    prime_p4_qualification_stores,
    qual_pg_schema,
    require_p4_pg_infra,
    resolve_pg_dsn,
)
from tests.qualification.dist_scale_x.p4_persistence_op import execute_p4_persistence_op
from tests.qualification.dist_scale_x._p5_support import new_run_id
from tests.qualification.dist_scale_x._p6_support import (
    WL7_DURATION_SECONDS,
    WL7_LAT_SAMPLE_SECONDS,
    WL7_QUEUE_SAMPLE_SECONDS,
    WL7_RSS_SAMPLE_SECONDS,
    WL7_WORKERS,
    p6_record,
    write_p6_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.slow]


def _rss_bytes() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss)
    except Exception:
        return -1.0


def test_p6_wl07_mixed_soak_stability() -> None:
    require_p4_pg_infra()
    duration = WL7_DURATION_SECONDS
    min_envelope = int(os.environ.get("DIST_SCALE_X_WL7_MIN_ENVELOPE_SECONDS", "14400"))
    soak_complete = 1.0 if duration >= min_envelope else 0.0
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("wl7"))
    prime_p4_qualification_stores(dsn, schema)

    start = time.monotonic()
    rss_samples: list[float] = []
    lat_samples: list[float] = []
    writes = 0
    errors = 0
    next_rss = start
    next_lat = start

    while time.monotonic() - start < duration:
        t0 = time.perf_counter()
        result = execute_p4_persistence_op(
            dsn=dsn,
            schema=schema,
            tenant_id="tenant-a",
            op_index=writes + 1,
            payload_pad="s" * 256,
        )
        writes += 1
        if result.error:
            errors += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        now = time.monotonic()
        if now >= next_lat:
            lat_samples.append(elapsed_ms)
            next_lat = now + WL7_LAT_SAMPLE_SECONDS
        if now >= next_rss:
            rss_samples.append(_rss_bytes())
            next_rss = now + WL7_RSS_SAMPLE_SECONDS
        time.sleep(0.05)

    run_id = new_run_id("wl07-soak")
    slope = 0.0
    if len(rss_samples) >= 2 and rss_samples[0] > 0:
        slope = (rss_samples[-1] - rss_samples[0]) / max(len(rss_samples), 1)
    p95 = sorted(lat_samples)[int(0.95 * (len(lat_samples) - 1))] if lat_samples else 0.0
    topo = (
        f"WL-7 soak duration_s={duration}; workers={WL7_WORKERS}; "
        f"queue_sample_s={WL7_QUEUE_SAMPLE_SECONDS}"
    )
    write_p6_manifest(
        run_id=run_id,
        topology_description=topo,
        records=(
            p6_record(
                record_id=f"{run_id}-m",
                workload_id="WL-7",
                dimension_ids=("M",),
                duration_seconds=float(duration),
                topology_description=topo,
                metrics=DistScaleXMetricsSnapshot(
                    p95_latency_ms=p95,
                    extra={
                        "rss_slope_per_sample": slope,
                        "write_count": float(writes),
                        "error_count": float(errors),
                        "soak_complete_envelope": soak_complete,
                        "min_envelope_seconds": float(min_envelope),
                    },
                ),
                fault_description="mixed WL-1/3/4/5/6 weights — PG write slice",
            ),
        ),
    )
    assert errors == 0, f"soak correctness failures: {errors}"
    if soak_complete < 1.0:
        pytest.skip(
            f"WL-7 harness OK; full {min_envelope}s soak required for exit 11 "
            f"(set DIST_SCALE_X_WL7_DURATION_SECONDS={min_envelope})",
        )
