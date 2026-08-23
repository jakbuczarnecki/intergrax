# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from intergrax.contracts.agent_step import AgentStep
from intergrax.contracts.evidence_claims import ClaimResolution
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.tool_executor import ToolHandler
from platform_proofs.scenarios.ai_incident_investigation.domain_reasoning import (
    parse_telemetry_payload,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INVESTIGATOR_AGENT_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import build_runtime_bundle
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    TOOL_TELEMETRY_READ,
    TelemetryInput,
    TelemetryOutput,
    default_telemetry_input,
)
from testing_support.builder import tools_agent_make_contract

pytestmark = pytest.mark.unit


def _build_runtime_state(bundle) -> RuntimeState:
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )
    ctx = bundle.investigator.build_context(request)
    return RuntimeState(
        context=ctx,
        request=request,
        run_id=request.run_id,
        tool_traces=[],
    )


def _investigator_step() -> AgentStep:
    return AgentStep(
        step_id="investigate",
        step_name="investigate",
        step_index=0,
        trace_label="incident_investigation.investigate",
        allowed_tools=[],
    )


class _AdversarialTelemetryHandler(ToolHandler[TelemetryInput, TelemetryOutput]):
    def __init__(self) -> None:
        self.call_count = 0

    def execute(self, request: ToolExecutionRequest[TelemetryInput]) -> TelemetryOutput:
        self.call_count += 1
        if self.call_count == 1:
            return TelemetryOutput(
                station_id=request.input.station_id,
                window=request.input.window,
                signal_state="intermittent_degraded",
                complex_assembly_throughput_pct=62.0,
                baseline_throughput_pct=91.0,
                observed_from="2026-01-01T08:00:00",
                observed_to="2026-01-01T09:00:00",
                admissible=True,
            )
        return TelemetryOutput(
            station_id=request.input.station_id,
            window=request.input.window,
            signal_state="healthy",
            complex_assembly_throughput_pct=90.0,
            baseline_throughput_pct=91.0,
            observed_from="2026-01-01T08:00:00",
            observed_to="2026-01-01T09:00:00",
            admissible=True,
        )


def _bundle_with_adversarial_telemetry():
    bundle = build_runtime_bundle()
    handler = _AdversarialTelemetryHandler()
    bundle.registry.unregister(TOOL_TELEMETRY_READ)
    bundle.registry.register(
        tools_agent_make_contract(TOOL_TELEMETRY_READ, TelemetryInput, TelemetryOutput),
        handler,
    )
    return bundle, handler


@pytest.mark.asyncio
async def test_invoke_tool_executes_provider_exactly_once() -> None:
    bundle, handler = _bundle_with_adversarial_telemetry()
    runtime_state = _build_runtime_state(bundle)
    step = _investigator_step()
    telemetry_input = default_telemetry_input(bundle.investigator._station_id)

    output = await bundle.investigator._invoke_tool(
        runtime_state=runtime_state,
        step=step,
        tool_id=TOOL_TELEMETRY_READ,
        tool_input=telemetry_input,
    )

    assert handler.call_count == 1
    assert len(runtime_state.tool_traces) == 1
    assert output["signal_state"] == "intermittent_degraded"


@pytest.mark.asyncio
async def test_gateway_output_matches_evidence_and_reasoning_payload() -> None:
    bundle, handler = _bundle_with_adversarial_telemetry()
    runtime_state = _build_runtime_state(bundle)
    step = _investigator_step()
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
        metadata={"critic_feedback": ["revise"]},
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=mint_attempt_id(),
        agent_id=INVESTIGATOR_AGENT_ID,
        request=request,
        metadata={"runtime_state": runtime_state},
    )
    captured: list[ToolResponse] = []
    original_invoke_request = ToolRuntime.invoke_request

    async def capture_invoke_request(**kwargs):
        response = await original_invoke_request(**kwargs)
        if kwargs["request"].tool_name == TOOL_TELEMETRY_READ:
            captured.append(response)
        return response

    with patch.object(ToolRuntime, "invoke_request", side_effect=capture_invoke_request):
        step_output = await bundle.investigator.run_step(step, exec_ctx)

    domain = step_output.data["domain_summary"]
    telemetry_node = next(
        node for node in domain["evidence_nodes"] if node["evidence_id"] == str(TELEMETRY_EVIDENCE_ID)
    )
    gateway_output = captured[-1].output

    assert handler.call_count == 1
    assert captured[-1].status is ToolResponseStatus.SUCCESS
    assert gateway_output is not None
    assert telemetry_node["payload"] == gateway_output
    parsed = parse_telemetry_payload(gateway_output)
    assert parsed.signal_state == "intermittent_degraded"
    h3_claim = next(c for c in domain["claim_set"]["claims"] if c["claim_id"] == str(REVISED_CLAIM_ID))
    assert h3_claim["resolution"] == ClaimResolution.SUPPORTED.value
    assert str(TELEMETRY_EVIDENCE_ID) in h3_claim["supporting_evidence_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error"),
    [
        (ToolResponseStatus.DENIED, "tool_not_allowed"),
        (ToolResponseStatus.TIMEOUT, "tool_timeout"),
        (ToolResponseStatus.FAILED, "tool_failed"),
    ],
)
async def test_non_success_tool_response_fails_without_evidence(
    status: ToolResponseStatus,
    error: str,
) -> None:
    bundle = build_runtime_bundle()
    runtime_state = _build_runtime_state(bundle)
    step = _investigator_step()
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=mint_attempt_id(),
        agent_id=INVESTIGATOR_AGENT_ID,
        request=request,
        metadata={"runtime_state": runtime_state},
    )

    failing_response = ToolResponse(
        request_id="tool_test_fail",
        status=status,
        error=error,
        duration_ms=1,
    )
    with patch.object(
        ToolRuntime,
        "invoke_request",
        new=AsyncMock(return_value=failing_response),
    ):
        with pytest.raises(RuntimeError, match="failed: status="):
            await bundle.investigator.run_step(step, exec_ctx)


@pytest.mark.asyncio
async def test_success_without_output_fails_without_evidence() -> None:
    bundle = build_runtime_bundle()
    runtime_state = _build_runtime_state(bundle)
    step = _investigator_step()

    missing_output = ToolResponse(
        request_id="tool_test_missing_output",
        status=ToolResponseStatus.SUCCESS,
        output=None,
        duration_ms=1,
    )
    with patch.object(
        ToolRuntime,
        "invoke_request",
        new=AsyncMock(return_value=missing_output),
    ):
        with pytest.raises(RuntimeError, match="output is missing"):
            await bundle.investigator._invoke_tool(
                runtime_state=runtime_state,
                step=step,
                tool_id=TOOL_TELEMETRY_READ,
                tool_input=default_telemetry_input(bundle.investigator._station_id),
            )


def test_investigator_has_no_direct_registry_executor_execution() -> None:
    import inspect

    from platform_proofs.scenarios.ai_incident_investigation import investigator_agent as mod

    source = inspect.getsource(mod)
    assert "_registry_executor.execute(" not in source
    assert "self._registry_executor" not in source
    assert "ToolExecutionRequest(" not in source
    assert "RegistryToolExecutor(" in source
    assert "ToolRuntime.invoke_request(" in source
    assert "ToolRuntime.invoke(" not in source.replace("ToolRuntime.invoke_request(", "")
