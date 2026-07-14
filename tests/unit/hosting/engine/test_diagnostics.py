# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.hosting.engine.diagnostics import (
    DiagnosticsRecorder,
    HostedApplicationFailurePhase,
    HostedApplicationOperationPhase,
)
from intergrax.hosting.engine.health import HostedApplicationHealthSnapshot
from intergrax.hosting.engine.lifecycle import HostedApplicationLifecycleController
from intergrax.hosting import HostedApplicationLifecycleState
from tests.unit.hosting.engine._fakes import FixedClock

pytestmark = pytest.mark.unit


def test_deterministic_failure_ids_with_injected_generator() -> None:
    clock = FixedClock()
    ids = iter(["failure-aaa", "failure-bbb"])

    def gen() -> str:
        return next(ids)

    recorder = DiagnosticsRecorder(
        clock=clock,
        application_id="test_app",
        instance_id="instance-001",
        profile_digest="sha256:" + "a" * 64,
        definition_digest="sha256:" + "b" * 64,
        failure_id_generator=gen,
    )
    recorder.record_primary_failure(
        phase=HostedApplicationFailurePhase.COMPONENT_START,
        source_kind="component",
        source_id="worker",
        exc=RuntimeError("boom"),
        reason_code="component_failed",
    )
    recorder.record_secondary_failure(
        phase=HostedApplicationFailurePhase.ROLLBACK,
        source_kind="cleanup",
        source_id="stop",
        exc=RuntimeError("secondary"),
        reason_code="rollback_failed",
    )
    lifecycle = HostedApplicationLifecycleController(clock)
    lifecycle.transition_to(HostedApplicationLifecycleState.STARTING, reason_code="starting")
    lifecycle.transition_to(HostedApplicationLifecycleState.FAILED, reason_code="failed")
    health = HostedApplicationHealthSnapshot(
        live=False,
        ready=False,
        degraded=False,
        accepting_new_work=False,
        runtime_ready=False,
        instance_ownership_valid=False,
        shutdown_requested=False,
        last_evaluated_at=clock.now(),
    )
    recorder.set_operation_phase(HostedApplicationOperationPhase.ROLLBACK)
    first = recorder.snapshot(lifecycle=lifecycle.snapshot(), health=health).model_dump(mode="json")
    second = recorder.snapshot(lifecycle=lifecycle.snapshot(), health=health).model_dump(mode="json")
    assert first["current_failure"]["failure_id"] == "failure-aaa"
    assert first["secondary_failures"][0]["failure_id"] == "failure-bbb"
    assert first["current_failure"]["phase"] == HostedApplicationFailurePhase.COMPONENT_START.value
    assert first["active_operation_phase"] == HostedApplicationOperationPhase.ROLLBACK.value
    assert first["snapshot_timestamp"] == second["snapshot_timestamp"]
