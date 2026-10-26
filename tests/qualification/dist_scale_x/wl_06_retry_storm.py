# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-6 retry storm (HTTP invoker + fault mix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p3_support import (
    WL6_DRAIN_CAP_SECONDS,
    WL6_FAULT_SECONDS,
    new_run_id,
    p3_record,
    require_p3_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from tests.qualification.dist_scale_x.p3_strict_http_stack import (
    run_concurrent_invocations,
    start_p3_strict_http_stack,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]


def test_p3_wl06_retry_storm_http_invoker_supplementary() -> None:
    require_p3_infra()
    run_id = new_run_id("wl06-storm")
    tmp = Path(__file__).resolve().parents[3] / ".tmp" / "session" / "dist-scale-x-p3-wl6" / run_id
    tmp.mkdir(parents=True, exist_ok=True)
    stack = start_p3_strict_http_stack(tmp, fault_mode="mix", slow_seconds=30.0)
    try:
        stats_fault = run_concurrent_invocations(
            stack,
            duration_s=WL6_FAULT_SECONDS,
            tenants=("tenant-a", "tenant-b"),
            concurrency=12,
        )
        stats_drain = run_concurrent_invocations(
            stack,
            duration_s=min(60.0, WL6_DRAIN_CAP_SECONDS),
            tenants=("tenant-a", "tenant-b"),
            concurrency=4,
        )
        total_attempts = stats_fault.physical_attempts + stats_drain.physical_attempts
        logical_ok = stats_fault.logical_success + stats_drain.logical_success
        submissions = max(logical_ok, 1)
        amp = total_attempts / float(submissions)
        assert amp < 50.0, f"retry amplification excessive: {amp}"
        topology = (
            f"P3 WL-6; fault_window_s={WL6_FAULT_SECONDS}; drain_cap_s={WL6_DRAIN_CAP_SECONDS}; "
            f"fault_mix=20% class via HTTP mix handler; supplementary HTTP slice (RabbitMQ M in wl_06_rabbit_broker_storm)"
        )
        record = p3_record(
            run_id=run_id,
            suffix="wl06-http",
            workload_id="WL-6",
            dimension_ids=("C", "E"),
            duration_seconds=WL6_FAULT_SECONDS + min(60.0, WL6_DRAIN_CAP_SECONDS),
            topology_description=topology,
            fault_description="20% transient HTTP mix",
            metrics=DistScaleXMetricsSnapshot(
                retry_amplification_factor=amp,
                extra={
                    "fault_logical_success": float(stats_fault.logical_success),
                    "fault_logical_failure": float(stats_fault.logical_failure),
                    "fault_http_requests": float(stats_fault.http_requests),
                    "drain_logical_success": float(stats_drain.logical_success),
                    "total_physical_attempts": float(total_attempts),
                },
            ),
        )
        write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
        assert stats_fault.logical_success + stats_fault.logical_failure > 0
    finally:
        stack.close()
