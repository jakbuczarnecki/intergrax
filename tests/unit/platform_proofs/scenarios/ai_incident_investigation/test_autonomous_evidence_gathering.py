# © Artur Czarnecki. All rights reserved.

"""APP-2A autonomous evidence gathering and decision observability tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.contracts.agent_step import AgentStep
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from pydantic import BaseModel
from platform_proofs.scenarios.ai_incident_investigation.evidence_gathering import (
    gather_incident_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.incident_scope import (
    IncidentScope,
    IncidentScopeViolationError,
)
from platform_proofs.scenarios.ai_incident_investigation.investigation_observability import (
    IncidentPlannerDecisionDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INVESTIGATOR_AGENT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    OUTCOME_RESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from tests.unit.platform_proofs.scenarios.ai_incident_investigation.planner_doubles import (
    ScriptedIncidentInvestigationLLM,
)

pytestmark = pytest.mark.unit


class _SentinelCanonicalInvoker:
    """Test double proving evidence gathering delegates to config.tool_invoker."""

    def __init__(self, *, inner: ToolInvokerProtocol, registry: ToolRegistry) -> None:
        self._inner = inner
        self._registry = registry
        self.invoke_count = 0

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def invoke(
        self,
        *,
        state: object,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> object:
        self.invoke_count += 1
        return self._inner.invoke(state=state, agent_id=agent_id, request=request)


class _MarkedDecoratorInvoker:
    """Decorator sentinel proving IncidentScope wrapper does not unwrap canonical invoker."""

    marker_id = "incident-canonical-decorator-sentinel"

    def __init__(self, inner: ToolInvokerProtocol) -> None:
        self._inner = inner
        self.wrapped_invocations = 0

    @property
    def registry(self) -> ToolRegistry:
        return self._inner.registry

    def invoke(
        self,
        *,
        state: object,
        agent_id: str,
        request: ToolExecutionRequest[BaseModel],
    ) -> object:
        self.wrapped_invocations += 1
        return self._inner.invoke(state=state, agent_id=agent_id, request=request)


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
    return RuntimeState(context=ctx, request=request, run_id=request.run_id)


def test_incident_scope_rejects_out_of_scope_line() -> None:
    scope = IncidentScope.from_fixture_defaults(station_id="complex_assembly_station")
    with pytest.raises(IncidentScopeViolationError, match="line_id_out_of_scope"):
        scope.validate_tool_input(
            TOOL_WORKLOAD_READ,
            {"line_id": "line_z", "window": scope.incident_window},
        )


def test_planner_decision_payload_schema_and_redaction() -> None:
    payload = IncidentPlannerDecisionDiagV1(
        round_index=1,
        investigation_phase="initial",
        objective="gather workload evidence",
        selected_tool_ids=(TOOL_WORKLOAD_READ,),
        evidence_basis_tool_call_ids=(),
        evidence_basis_evidence_ids=(),
        incident_line_id="line4",
        incident_station_id="complex_assembly_station",
    )
    assert payload.schema_id() == "incident.planner_decision.v1"
    redacted = payload.redact()
    assert redacted.objective == "[REDACTED]"
    assert redacted.selected_tool_ids == (TOOL_WORKLOAD_READ,)


@pytest.mark.asyncio
async def test_alternative_tool_order_executes_planner_selected_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alt_order = (
        TOOL_COMPARISON_READ,
        TOOL_WORKLOAD_READ,
        TOOL_THROUGHPUT_READ,
        TOOL_TELEMETRY_READ,
        TOOL_STAFFING_ATTENDANCE_READ,
        TOOL_STAFFING_SCHEDULE_READ,
    )
    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        lambda *_args, **_kwargs: ScriptedIncidentInvestigationLLM(
            initial_sequence=alt_order[:3],
            revision_sequence=alt_order,
        ),
    )
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    assert result.tool_invocations >= 6

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
    runtime_state = bundle.investigator.build_context(request)
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

    state = RuntimeState(context=runtime_state, request=request, run_id=request.run_id)
    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
        is_revision=True,
        critic_feedback=["unsupported inference"],
    )
    assert gathering.tool_execution_order[: len(alt_order)] == alt_order


@pytest.mark.asyncio
async def test_observability_correlates_decision_trace_tool_trace_and_evidence() -> None:
    bundle = build_runtime_bundle()
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )
    runtime_state = bundle.investigator.build_context(request)
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

    state = RuntimeState(context=runtime_state, request=request, run_id=request.run_id)
    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
        is_revision=False,
    )
    planner_events = [
        event
        for event in state.trace_events
        if event.component is TraceComponent.PLANNER
        and event.step == "incident_planner_decision"
    ]
    assert len(planner_events) >= 2
    first = planner_events[0].payload
    assert isinstance(first, IncidentPlannerDecisionDiagV1)
    assert first.selected_tool_ids
    assert first.objective
    assert state.tool_traces[0].tool_name == first.selected_tool_ids[0]
    created_id = gathering.evidence_nodes[0]["evidence_id"]
    assert created_id
    second = planner_events[1].payload
    assert isinstance(second, IncidentPlannerDecisionDiagV1)
    if second.evidence_basis_evidence_ids:
        assert created_id in second.evidence_basis_evidence_ids or second.round_index > 1


def test_canonical_investigator_has_no_hardcoded_three_plus_three_sequence() -> None:
    from platform_proofs.scenarios.ai_incident_investigation import investigator_agent as mod

    source = inspect.getsource(mod.IncidentInvestigatorAgent.run_step)
    assert "TOOL_WORKLOAD_READ" not in source
    assert "TOOL_THROUGHPUT_READ" not in source
    assert "default_line_window_input" not in source
    assert "gather_incident_evidence(" in source


@pytest.mark.asyncio
async def test_scope_violation_emits_diagnostic_without_unresolved_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OutOfScopeLLM(ScriptedIncidentInvestigationLLM):
        def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
            response = super().generate_with_tools(messages, tools_schema, **kwargs)
            if not response.tool_calls:
                return response
            original = response.tool_calls[0]
            return LLMAdapterResponse(
                content=response.content,
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id=original.id,
                        name=TOOL_WORKLOAD_READ,
                        arguments={"line_id": "line_z", "window": "incident_window"},
                    ),
                ),
            )

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        lambda *_args, **_kwargs: _OutOfScopeLLM(initial_sequence=(TOOL_WORKLOAD_READ,)),
    )
    bundle = build_runtime_bundle()
    request = RuntimeRequest(
        agent_id=INVESTIGATOR_AGENT_ID,
        user_id="u",
        session_id="s",
        tenant_id="t",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        message="investigate",
    )
    runtime_state = bundle.investigator.build_context(request)
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

    state = RuntimeState(context=runtime_state, request=request, run_id=request.run_id)
    step = AgentStep(
        step_id="investigate",
        step_name="investigate",
        step_index=0,
        trace_label="incident_investigation.investigate",
        allowed_tools=[],
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=mint_attempt_id(),
        agent_id=INVESTIGATOR_AGENT_ID,
        request=request,
        metadata={"runtime_state": state},
    )
    with pytest.raises(IncidentScopeViolationError):
        await bundle.investigator.run_step(step, exec_ctx)
    assert any(event.step == "incident_scope_rejection" for event in state.trace_events)


def test_gather_incident_evidence_requires_runtime_config_tool_invoker() -> None:
    bundle = build_runtime_bundle()
    state = _build_runtime_state(bundle)
    state.context.config.tool_invoker = None
    with pytest.raises(RuntimeError, match="incident_runtime_tool_invoker_missing"):
        gather_incident_evidence(
            runtime_state=state,
            registry=bundle.registry,
            scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
            is_revision=False,
        )


def test_gather_incident_evidence_delegates_to_runtime_config_tool_invoker() -> None:
    bundle = build_runtime_bundle()
    state = _build_runtime_state(bundle)
    canonical = state.context.config.tool_invoker
    assert canonical is not None
    sentinel = _SentinelCanonicalInvoker(inner=canonical, registry=bundle.registry)
    state.context.config.tool_invoker = sentinel

    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
        is_revision=False,
    )

    assert sentinel.invoke_count == gathering.tool_invocations
    assert sentinel.invoke_count > 0


def test_incident_scoped_invoker_preserves_decorated_canonical_invoker() -> None:
    bundle = build_runtime_bundle()
    state = _build_runtime_state(bundle)
    canonical = state.context.config.tool_invoker
    assert canonical is not None
    decorated = _MarkedDecoratorInvoker(canonical)
    state.context.config.tool_invoker = decorated

    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
        is_revision=False,
    )

    assert decorated.wrapped_invocations == gathering.tool_invocations
    assert decorated.wrapped_invocations > 0


def test_scope_rejection_does_not_invoke_canonical_tool_invoker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OutOfScopeLLM(ScriptedIncidentInvestigationLLM):
        def generate_with_tools(self, messages, tools_schema, **kwargs):  # type: ignore[no-untyped-def]
            response = super().generate_with_tools(messages, tools_schema, **kwargs)
            if not response.tool_calls:
                return response
            original = response.tool_calls[0]
            return LLMAdapterResponse(
                content=response.content,
                tool_calls=(
                    LLMToolCall.from_openai_shape(
                        call_id=original.id,
                        name=TOOL_WORKLOAD_READ,
                        arguments={"line_id": "line_z", "window": "incident_window"},
                    ),
                ),
            )

    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.runtime_composition.resolve_llm_adapter",
        lambda *_args, **_kwargs: _OutOfScopeLLM(initial_sequence=(TOOL_WORKLOAD_READ,)),
    )
    bundle = build_runtime_bundle()
    state = _build_runtime_state(bundle)
    canonical = state.context.config.tool_invoker
    assert canonical is not None
    sentinel = _SentinelCanonicalInvoker(inner=canonical, registry=bundle.registry)
    state.context.config.tool_invoker = sentinel

    with pytest.raises(IncidentScopeViolationError):
        gather_incident_evidence(
            runtime_state=state,
            registry=bundle.registry,
            scope=IncidentScope.from_fixture_defaults(station_id=bundle.fixture.telemetry.station_id),
            is_revision=False,
        )

    assert sentinel.invoke_count == 0
    assert any(event.step == "incident_scope_rejection" for event in state.trace_events)


def test_evidence_gathering_has_no_local_runtime_tool_invoker_construction() -> None:
    from platform_proofs.scenarios.ai_incident_investigation import evidence_gathering as mod

    source = inspect.getsource(mod)
    assert "RegistryToolExecutor(" not in source
    assert "RuntimeToolInvoker(" not in source
