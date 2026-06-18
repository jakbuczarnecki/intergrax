# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.attestation.attestation_policy import (
    AttestationCaptureMode,
    should_emit_boundary_event,
)
from intergrax.runtime.attestation.buffer import BoundaryEventBuffer
from intergrax.runtime.attestation.boundary_emitter import ExecutionBoundaryEmitter
from intergrax.runtime.attestation.execution_boundary_event import ExecutionBoundaryEventV1
from intergrax.runtime.attestation.settings import (
    ExecutionBoundaryExportRuntimeSettings,
    resolve_execution_boundary_export_runtime,
)
from intergrax.applications.contracts.environment_profile import ExecutionBoundaryExportProfile
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.providers.records.contracts import RecordsPutInput, RecordsPutOutput
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def test_should_emit_side_effects_only():
    contract = ToolContract(
        tool_id="records.put",
        name="records.put",
        description="put",
        input_schema=RecordsPutInput,
        output_schema=RecordsPutOutput,
        error_mapping={},
        side_effects=True,
        risk_level=ToolRiskLevel.MEDIUM,
    )
    result = ToolExecutionResult.ok(RecordsPutOutput(partition_key="a", row_key="b"))
    assert should_emit_boundary_event(
        contract=contract,
        result=result,
        capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
        allowlist=frozenset(),
    )
    read_only = ToolContract(
        tool_id="records.get",
        name="records.get",
        description="get",
        input_schema=RecordsPutInput,
        output_schema=RecordsPutOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
    )
    assert not should_emit_boundary_event(
        contract=read_only,
        result=result,
        capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
        allowlist=frozenset(),
    )


def test_resolve_execution_boundary_export_profile():
    profile = ExecutionBoundaryExportProfile(
        enabled=True,
        capture_mode="side_effects_only",
        step_level_enabled=True,
    )
    resolved = resolve_execution_boundary_export_runtime(profile)
    assert resolved is not None
    assert resolved.enabled is True
    assert resolved.step_level_enabled is True
    assert resolved.capture_mode == AttestationCaptureMode.SIDE_EFFECTS_ONLY


def test_boundary_event_buffer_append_and_snapshot():
    buffer = BoundaryEventBuffer()
    event = ExecutionBoundaryEventV1(
        event_id="evt-1",
        event_sequence=1,
        boundary_type="tool_execution",
        tool_id="records.put",
        agent_id="boundary_demo_agent",
        run_id="run-1",
        step_id="step-1",
        action_status="executed",
        side_effects=True,
        risk_level="medium",
        occurred_at="2026-06-13T12:00:00+00:00",
        lineage={"ref": "run-1:step-1", "type": "execution_record"},
    )
    buffer.append("run-1", event)
    snapshot = buffer.snapshot_for_run("run-1")
    assert len(snapshot) == 1
    assert snapshot[0]["signed"] is False


def test_execution_boundary_emitter_writes_failed_status_to_buffer() -> None:
    buffer = BoundaryEventBuffer()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        enable_rag=False,
        production_mode=False,
        execution_boundary_export=ExecutionBoundaryExportRuntimeSettings(
            enabled=True,
            capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
            allowlist=frozenset(),
        ),
        boundary_event_buffer=buffer,
    )
    runtime_context = RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
    )
    request = RuntimeRequest(
        tenant_id="lab",
        user_id="u1",
        session_id="s1",
        agent_id="boundary_demo_agent",
        message="hi",
        metadata={"run_id": "run_failed_emit", "task_id": "task_failed_emit"},
    )
    state = RuntimeState(context=runtime_context, request=request, run_id="run_failed_emit")
    contract = ToolContract(
        tool_id="records.put",
        name="records.put",
        description="put",
        input_schema=RecordsPutInput,
        output_schema=RecordsPutOutput,
        error_mapping={},
        side_effects=True,
        risk_level=ToolRiskLevel.MEDIUM,
    )
    tool_request = ToolExecutionRequest(
        run_id="run_failed_emit",
        step_id="store_demo_record",
        tool_id="records.put",
        input=RecordsPutInput(partition_key="p", row_key="r", data={"title": "x"}),
    )
    from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode

    result = ToolExecutionResult.fail(RuntimeErrorCode.VALIDATION_ERROR, "validation failed")
    ExecutionBoundaryEmitter.maybe_emit(
        state=state,
        agent_id="boundary_demo_agent",
        contract=contract,
        request=tool_request,
        result=result,
    )
    events = buffer.snapshot_for_run("run_failed_emit")
    assert len(events) == 1
    assert events[0]["event_sequence"] == 1
    assert events[0]["boundary_type"] == "tool_execution"
    assert events[0]["action_status"] == "failed"
    assert events[0]["error_message"] == "validation failed"


def test_execution_boundary_emitter_writes_to_buffer():
    buffer = BoundaryEventBuffer()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        enable_rag=False,
        production_mode=False,
        execution_boundary_export=ExecutionBoundaryExportRuntimeSettings(
            enabled=True,
            capture_mode=AttestationCaptureMode.SIDE_EFFECTS_ONLY,
            allowlist=frozenset(),
        ),
        boundary_event_buffer=buffer,
    )
    runtime_context = RuntimeContext.build(
        config=config,
        session_manager=SessionManager(storage=InMemorySessionStorage()),
    )
    request = RuntimeRequest(
        tenant_id="lab",
        user_id="u1",
        session_id="s1",
        agent_id="boundary_demo_agent",
        message="hi",
        metadata={"run_id": "run_emit_1", "task_id": "task_emit_1"},
    )
    state = RuntimeState(context=runtime_context, request=request, run_id="run_emit_1")
    contract = ToolContract(
        tool_id="records.put",
        name="records.put",
        description="put",
        input_schema=RecordsPutInput,
        output_schema=RecordsPutOutput,
        error_mapping={},
        side_effects=True,
        risk_level=ToolRiskLevel.MEDIUM,
    )
    tool_request = ToolExecutionRequest(
        run_id="run_emit_1",
        step_id="store_demo_record",
        tool_id="records.put",
        input=RecordsPutInput(partition_key="p", row_key="r", data={"title": "x"}),
    )
    result = ToolExecutionResult.ok(
        RecordsPutOutput(stored=True, partition_key="p", row_key="r"),
    )
    ExecutionBoundaryEmitter.maybe_emit(
        state=state,
        agent_id="boundary_demo_agent",
        contract=contract,
        request=tool_request,
        result=result,
    )
    events = buffer.snapshot_for_run("run_emit_1")
    assert len(events) == 1
    assert events[0]["event_sequence"] == 1
    assert events[0]["boundary_type"] == "tool_execution"
    assert events[0]["tool_id"] == "records.put"
    assert events[0]["input"]["partition_key"] == "p"
