# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_runtime_bundle

import inspect

import pytest

from intergrax.contracts.agent_step import AgentStep
from intergrax.contracts.evidence_claims import ClaimResolution
import intergrax.contracts.execution_identity as execution_identity
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    parse_telemetry_payload,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    INVESTIGATOR_AGENT_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import TOOL_TELEMETRY_READ

pytestmark = pytest.mark.unit


def _build_runtime_state(bundle):
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=execution_identity.mint_task_id(),
        run_id=execution_identity.mint_run_id(),
        message="investigate",
    )
    ctx = bundle.investigator.build_context(request)
    execution_identity.bind_active_execution_identity(
        run_id=request.run_id,
        attempt_id=execution_identity.mint_attempt_id(),
        execution_id=execution_identity.mint_execution_id(),
    )
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

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


@pytest.mark.asyncio
async def test_run_step_executes_telemetry_once_via_tool_runtime() -> None:
    bundle = build_runtime_bundle()
    runtime_state = _build_runtime_state(bundle)
    step = _investigator_step()
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=execution_identity.mint_task_id(),
        run_id=execution_identity.mint_run_id(),
        message="investigate",
        metadata={"critic_feedback": ["revise"]},
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=execution_identity.mint_attempt_id(),
        agent_id=INVESTIGATOR_AGENT_ID,
        request=request,
        metadata={"runtime_state": runtime_state},
    )

    step_output = await bundle.investigator.run_step(step, exec_ctx)

    domain = step_output.data["domain_summary"]
    telemetry_node = next(
        node for node in domain["evidence_nodes"] if node["evidence_id"] == str(TELEMETRY_EVIDENCE_ID)
    )
    telemetry_traces = [
        trace for trace in runtime_state.tool_traces if trace.tool_name == TOOL_TELEMETRY_READ
    ]

    assert len(telemetry_traces) == 1
    assert telemetry_node["payload"]["availability"] == "available"
    parsed = parse_telemetry_payload(telemetry_node["payload"])
    assert parsed.signal_state == "intermittent_degraded"
    bindings = domain.get("claim_hypothesis_bindings", [])
    h3_claim_id = next(
        binding["claim_id"] for binding in bindings if binding.get("hypothesis_id") == "H3"
    )
    h3_claim = next(
        claim for claim in domain["claim_set"]["claims"] if claim["claim_id"] == h3_claim_id
    )
    assert h3_claim["resolution"] == ClaimResolution.PENDING.value
    assert str(TELEMETRY_EVIDENCE_ID) in h3_claim["supporting_evidence_ids"]


def test_investigator_has_no_direct_registry_executor_execution() -> None:
    from platform_proofs.scenarios.ai_incident_investigation.application import investigator_agent as mod

    source = inspect.getsource(mod)
    assert "_registry_executor.execute(" not in source
    assert "self._registry_executor" not in source
    assert "ToolExecutionRequest(" not in source
    assert "build_agent_runtime_context(" in source
    assert "gather_incident_evidence(" in source
    assert "SyntheticTelemetryProvider" not in source
