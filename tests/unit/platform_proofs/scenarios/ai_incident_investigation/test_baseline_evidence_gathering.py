# © Artur Czarnecki. All rights reserved.

"""Deterministic baseline incident evidence acquisition tests (DS-E2E-12)."""

from __future__ import annotations

import pytest

from intergrax.contracts import execution_identity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.active_execution_budget import bind_root_execution_budget
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.governance.active_execution_authority import bind_active_execution_authority
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_gathering import (
    gather_incident_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_scope import (
    IncidentScope,
)
from platform_proofs.scenarios.ai_incident_investigation.application.observability import (
    IncidentBaselineEvidenceDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    INVESTIGATOR_AGENT_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    build_runtime_bundle,
)
from tests.unit.platform_proofs.scenarios.ai_incident_investigation.planner_doubles import (
    ScriptedIncidentInvestigationLLM,
)

pytestmark = pytest.mark.unit


def _build_runtime_state(bundle) -> tuple[RuntimeState, execution_identity.ExecutionId]:
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
    execution_id = execution_identity.mint_execution_id()
    attempt_id = execution_identity.mint_attempt_id()
    execution_identity.bind_active_execution_identity(
        run_id=request.run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    bind_active_execution_authority(ParentExecutionAuthority.unrestricted_root())
    bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    return RuntimeState(context=ctx, request=request, run_id=request.run_id), execution_id


def _patch_planner(monkeypatch: pytest.MonkeyPatch, llm_factory) -> None:
    monkeypatch.setattr(
        "platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition.resolve_llm_adapter",
        lambda *_args, **_kwargs: llm_factory(),
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        lambda *_args, **_kwargs: llm_factory(),
    )


def test_baseline_invokes_staffing_schedule_when_planner_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _omit_staffing_llm():
        return ScriptedIncidentInvestigationLLM(
            initial_sequence=(TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ),
        )

    _patch_planner(monkeypatch, _omit_staffing_llm)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    scope = IncidentScope.from_operational_defaults(station_id=bundle.operational_data.station_id)

    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=scope,
        is_revision=False,
    )

    evidence_ids = {
        str(node.get("evidence_id"))
        for node in gathering.evidence_nodes
        if node.get("evidence_id")
    }
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in evidence_ids
    assert gathering.tool_execution_order.count(TOOL_STAFFING_SCHEDULE_READ) == 1
    baseline_events = [
        event
        for event in state.trace_events
        if event.step == "incident_baseline_evidence_acquisition"
    ]
    assert baseline_events
    payload = baseline_events[0].payload
    assert isinstance(payload, IncidentBaselineEvidenceDiagV1)
    assert TOOL_STAFFING_SCHEDULE_READ in payload.selected_tool_ids


def test_baseline_does_not_duplicate_planner_gathered_staffing_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _includes_staffing_llm():
        return ScriptedIncidentInvestigationLLM(
            initial_sequence=(
                TOOL_WORKLOAD_READ,
                TOOL_THROUGHPUT_READ,
                TOOL_STAFFING_SCHEDULE_READ,
            ),
        )

    _patch_planner(monkeypatch, _includes_staffing_llm)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    scope = IncidentScope.from_operational_defaults(station_id=bundle.operational_data.station_id)

    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=scope,
        is_revision=False,
    )

    assert gathering.tool_execution_order.count(TOOL_STAFFING_SCHEDULE_READ) == 1


def test_baseline_evidence_budget_exhaustion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _minimal_llm():
        return ScriptedIncidentInvestigationLLM(initial_sequence=())

    _patch_planner(monkeypatch, _minimal_llm)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(RunBudget(max_tool_calls=2)),
    )
    scope = IncidentScope.from_operational_defaults(station_id=bundle.operational_data.station_id)

    with pytest.raises(RuntimeError, match="incident_evidence_gathering_budget_exceeded"):
        gather_incident_evidence(
            runtime_state=state,
            registry=bundle.registry,
            scope=scope,
            is_revision=False,
        )


class _StaffingScheduleFailingInvoker:
    def __init__(self, inner: ToolInvokerProtocol, registry: ToolRegistry) -> None:
        self._inner = inner
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def invoke(self, state: RuntimeState, request: object, agent_id: str | None = None) -> object:
        if (
            isinstance(request, ToolExecutionRequest)
            and request.tool_id == TOOL_STAFFING_SCHEDULE_READ
        ):
            return ToolExecutionResult.fail("tool_failure", "staffing schedule unavailable")
        return self._inner.invoke(state=state, request=request, agent_id=agent_id)


def test_staffing_schedule_tool_failure_does_not_fabricate_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _omit_staffing_llm():
        return ScriptedIncidentInvestigationLLM(
            initial_sequence=(TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ),
        )

    _patch_planner(monkeypatch, _omit_staffing_llm)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    canonical = state.context.config.tool_invoker
    assert canonical is not None
    state.context.config.tool_invoker = _StaffingScheduleFailingInvoker(
        inner=canonical,
        registry=bundle.registry,
    )
    scope = IncidentScope.from_operational_defaults(station_id=bundle.operational_data.station_id)

    gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=scope,
        is_revision=False,
    )

    evidence_ids = {
        str(node.get("evidence_id"))
        for node in gathering.evidence_nodes
        if node.get("evidence_id")
    }
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) not in evidence_ids
    assert str(WORKLOAD_EVIDENCE_ID) in evidence_ids


def test_baseline_gathering_is_provider_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _omit_staffing_llm_a():
        return ScriptedIncidentInvestigationLLM(
            initial_sequence=(TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ),
        )

    _patch_planner(monkeypatch, _omit_staffing_llm_a)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    scope = IncidentScope.from_operational_defaults(station_id=bundle.operational_data.station_id)
    first_gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=scope,
        is_revision=False,
    )

    def _omit_staffing_llm_b():
        return ScriptedIncidentInvestigationLLM(
            initial_sequence=(TOOL_THROUGHPUT_READ, TOOL_WORKLOAD_READ),
        )

    _patch_planner(monkeypatch, _omit_staffing_llm_b)
    bundle = build_runtime_bundle()
    state, execution_id = _build_runtime_state(bundle)
    second_gathering = gather_incident_evidence(
        runtime_state=state,
        registry=bundle.registry,
        scope=scope,
        is_revision=False,
    )

    first_ids = {
        str(node.get("evidence_id"))
        for node in first_gathering.evidence_nodes
        if node.get("evidence_id")
    }
    second_ids = {
        str(node.get("evidence_id"))
        for node in second_gathering.evidence_nodes
        if node.get("evidence_id")
    }
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in first_ids
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in second_ids
    assert first_gathering.tool_execution_order.count(TOOL_STAFFING_SCHEDULE_READ) == 1
    assert second_gathering.tool_execution_order.count(TOOL_STAFFING_SCHEDULE_READ) == 1
