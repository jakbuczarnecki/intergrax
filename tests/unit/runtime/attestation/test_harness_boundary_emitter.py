# © Artur Czarnecki. All rights reserved.

import pytest
from types import SimpleNamespace

from intergrax.contracts.agent_run_enums import StepNextAction
from intergrax.contracts.agent_run_trace import AgentStepRecord, AgentStepStatus, PolicyCheckPhase, PolicyVerdictRecord
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.step_execution import StepExecutionRecord
from intergrax.runtime.attestation.attestation_policy import AttestationCaptureMode
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.harness_boundary_emitter import HarnessBoundaryEmitter
from intergrax.runtime.attestation.settings import ExecutionBoundaryExportRuntimeSettings

pytestmark = pytest.mark.unit


def _kernel_ctx(buffer: BoundaryEventBuffer, *, run_id: str = "run_harness_emit") -> SimpleNamespace:
    return SimpleNamespace(
        agent_id="boundary_demo_agent",
        run_id=run_id,
        task_id="task_harness_emit",
        tenant_id="lab",
        execution_boundary_export=ExecutionBoundaryExportRuntimeSettings(
            enabled=True,
            capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
            allowlist=frozenset(),
            step_level_enabled=True,
        ),
        boundary_event_buffer=buffer,
    )


def test_boundary_event_buffer_assigns_monotonic_event_sequence() -> None:
    from intergrax.runtime.attestation.execution_boundary_event import (
        ExecutionBoundaryEventV1,
        ExecutionBoundaryLineageV1,
    )

    buffer = BoundaryEventBuffer()
    base = dict(
        event_sequence=0,
        boundary_type="tool_execution",
        tool_id="records.put",
        agent_id="boundary_demo_agent",
        run_id="run-seq",
        step_id="store_demo_record",
        action_status="executed",
        side_effects=True,
        risk_level="medium",
        occurred_at="2026-06-13T12:00:00+00:00",
        lineage=ExecutionBoundaryLineageV1(ref="run-seq:store_demo_record"),
    )
    buffer.append("run-seq", ExecutionBoundaryEventV1(**{**base, "event_id": "evt-a"}))
    harness_payload = {
        **base,
        "event_id": "evt-b",
        "boundary_type": "harness_step",
        "tool_id": None,
        "side_effects": None,
        "risk_level": None,
        "action_status": "completed",
        "lineage": ExecutionBoundaryLineageV1(ref="run-seq:store_demo_record:harness_step"),
    }
    buffer.append("run-seq", ExecutionBoundaryEventV1(**harness_payload))
    snapshot = buffer.snapshot_for_run("run-seq")
    assert [row["event_sequence"] for row in snapshot] == [1, 2]
    assert snapshot[0]["boundary_type"] == "tool_execution"
    assert snapshot[1]["boundary_type"] == "harness_step"


def test_harness_boundary_emitter_writes_harness_step_event() -> None:
    buffer = BoundaryEventBuffer()
    kernel_ctx = _kernel_ctx(buffer)
    step_ctx = AgentStepContext(
        step_index=0,
        run_id="run_harness_emit",
        agent_id="boundary_demo_agent",
        contract_id="boundary_demo_agent",
        metadata={"step_id": "store_demo_record"},
    )
    record = StepExecutionRecord(
        step_index=0,
        outcome_applied=True,
        policy_pre=PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="allowed",
            policy_rule_id="kernel.default_allow",
        ),
        step_record=AgentStepRecord(
            step_id="store_demo_record",
            step_index=0,
            status=AgentStepStatus.SUCCEEDED,
            next_action=StepNextAction.CONTINUE,
            state_version=1,
            policy_verdicts=[
                PolicyVerdictRecord(
                    phase=PolicyCheckPhase.PRE,
                    action=PolicyAction.ALLOW,
                    reason="allowed",
                    policy_rule_id="kernel.default_allow",
                )
            ],
        ),
    )
    HarnessBoundaryEmitter.maybe_emit(
        kernel_ctx=kernel_ctx,
        step_ctx=step_ctx,
        record=record,
    )
    events = buffer.snapshot_for_run("run_harness_emit")
    assert len(events) == 1
    assert events[0]["boundary_type"] == "harness_step"
    assert events[0]["event_sequence"] == 1
    assert events[0]["step_id"] == "store_demo_record"
    assert events[0]["step_outcome"]["status"] == "completed"
    assert events[0]["policy_verdicts"][0]["phase"] == "pre"


def test_harness_boundary_emitter_skipped_when_step_level_disabled() -> None:
    buffer = BoundaryEventBuffer()
    kernel_ctx = _kernel_ctx(buffer, run_id="run_disabled")
    kernel_ctx.execution_boundary_export = ExecutionBoundaryExportRuntimeSettings(
        enabled=True,
        capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
        allowlist=frozenset(),
        step_level_enabled=False,
    )
    step_ctx = AgentStepContext(
        step_index=0,
        run_id="run_disabled",
        agent_id="boundary_demo_agent",
        contract_id="boundary_demo_agent",
    )
    record = StepExecutionRecord(step_index=0, outcome_applied=True)
    HarnessBoundaryEmitter.maybe_emit(
        kernel_ctx=kernel_ctx,
        step_ctx=step_ctx,
        record=record,
    )
    assert buffer.snapshot_for_run("run_disabled") == []
