# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-4 HTTP loopback physical (RuntimeToolInvoker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p3_support import (
    WL4_SCENARIO_SECONDS,
    new_run_id,
    p3_record,
    require_p3_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from tests.qualification.dist_scale_x.p3_strict_http_stack import (
    invoke_idempotent_post,
    run_concurrent_invocations,
    start_p3_strict_http_stack,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]


def _run_scenario(
    tmp_path: Path,
    *,
    run_prefix: str,
    fault_mode: str,
    workload_label: str,
    dimension_ids: tuple[str, ...],
    duration_s: float,
    concurrency: int = 8,
) -> None:
    require_p3_infra()
    run_id = new_run_id(run_prefix)
    stack = start_p3_strict_http_stack(
        tmp_path / run_id,
        fault_mode=fault_mode,  # type: ignore[arg-type]
        slow_seconds=30.0,
    )
    try:
        stats = run_concurrent_invocations(
            stack,
            duration_s=duration_s,
            tenants=("tenant-a", "tenant-b"),
            concurrency=concurrency,
        )
        amp = (
            stats.physical_attempts / stats.logical_success
            if stats.logical_success > 0
            else 0.0
        )
        topology = (
            f"P3 WL-4 HTTP; fault_mode={fault_mode}; duration_s={duration_s}; "
            f"invoker=RuntimeToolInvoker+declarative; loopback=127.0.0.1"
        )
        record = p3_record(
            run_id=run_id,
            suffix=workload_label,
            workload_id="WL-4",
            dimension_ids=dimension_ids,
            duration_seconds=duration_s,
            topology_description=topology,
            fault_description=fault_mode,
            metrics=DistScaleXMetricsSnapshot(
                throughput_per_sec=stats.logical_success / duration_s if duration_s else 0.0,
                retry_amplification_factor=amp,
                extra={
                    "http_requests": float(stats.http_requests),
                    "logical_success": float(stats.logical_success),
                    "logical_failure": float(stats.logical_failure),
                    "physical_attempts": float(stats.physical_attempts),
                },
            ),
        )
        write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
        assert stats.logical_success > 0 or stats.logical_failure > 0
    finally:
        stack.close()


@pytest.mark.parametrize(
    ("fault_mode", "label"),
    [
        ("ok", "ok"),
        ("503", "http-503"),
        ("slow", "slow-deadline"),
        ("mix", "fault-mix"),
    ],
)
def test_p3_wl04_http_scenario_classes(tmp_path: Path, fault_mode: str, label: str) -> None:
    _run_scenario(
        tmp_path,
        run_prefix=f"wl04-{label}",
        fault_mode=fault_mode,
        workload_label=label,
        dimension_ids=("A", "E"),
        duration_s=WL4_SCENARIO_SECONDS,
    )


def test_p3_wl04_strict_host_mse_denies_without_authorization(tmp_path: Path) -> None:
    from intergrax.runtime.agent_governance.errors import ToolGovernanceDeniedError

    require_p3_infra()
    run_id = new_run_id("wl04-mse-deny")
    stack = start_p3_strict_http_stack(tmp_path / run_id, fault_mode="ok", mse_allow=False)
    try:
        with pytest.raises(ToolGovernanceDeniedError, match="test-mse"):
            invoke_idempotent_post(
                stack,
                tenant_id="tenant-a",
                idempotency_key="mse-deny-key",
                body='{"probe":1}',
            )
    finally:
        stack.close()


def test_p3_wl04_lost_ack_single_logical_effect(tmp_path: Path) -> None:
    require_p3_infra()
    run_id = new_run_id("wl04-lost-ack")
    stack = start_p3_strict_http_stack(tmp_path / run_id, fault_mode="lost_ack")
    try:
        key = "lost-ack-key-1"
        body = '{"charge":1}'
        from intergrax.contracts.idempotency_store import InvocationUncertaintyError

        ok1 = invoke_idempotent_post(stack, tenant_id="tenant-a", idempotency_key=key, body=body)
        effects_after_first = stack.oracle.effect_count(key)
        try:
            invoke_idempotent_post(stack, tenant_id="tenant-a", idempotency_key=key, body=body)
            ok2 = True
        except InvocationUncertaintyError:
            ok2 = False
        effects = stack.oracle.effect_count(key)
        assert effects <= 1
        assert effects_after_first <= 1
        topology = "P3 WL-4 lost_ack_after_commit; single idempotency key"
        record = p3_record(
            run_id=run_id,
            suffix="lost-ack",
            workload_id="WL-4",
            dimension_ids=("E",),
            duration_seconds=1.0,
            topology_description=topology,
            fault_description="lost_ack_after_commit",
            metrics=DistScaleXMetricsSnapshot(
                extra={
                    "effect_count": float(effects),
                    "first_invoke_ok": float(1 if ok1 else 0),
                    "second_invoke_uncertain_blocked": float(0 if ok2 else 1),
                },
            ),
        )
        write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
    finally:
        stack.close()
