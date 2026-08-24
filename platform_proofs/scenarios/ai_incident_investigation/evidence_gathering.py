# © Artur Czarnecki. All rights reserved.

"""Autonomous incident evidence gathering via canonical bounded tool loop (APP-2A)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.registry_tool_executor import RegistryToolExecutor
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.incident_scope import (
    IncidentScope,
    IncidentScopeViolationError,
)
from platform_proofs.scenarios.ai_incident_investigation.investigation_observability import (
    IncidentPlannerDecisionDiagV1,
    IncidentPlannerStopDiagV1,
    IncidentScopeRejectionDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import (
    SCENARIO_TOOL_IDS,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_READ,
)

MAX_INCIDENT_TOOL_LOOP_ITERATIONS = 12
INCIDENT_INVESTIGATION_POLICY_PROMPT_ID = "incident_investigation_policy"

_TOOL_EVIDENCE_MAP: dict[str, tuple[str, str]] = {
    TOOL_WORKLOAD_READ: (str(WORKLOAD_EVIDENCE_ID), "workload observation"),
    TOOL_THROUGHPUT_READ: (str(THROUGHPUT_EVIDENCE_ID), "throughput observation"),
    TOOL_STAFFING_SCHEDULE_READ: (
        str(STAFFING_PRELIMINARY_EVIDENCE_ID),
        "staffing schedule observation",
    ),
    TOOL_STAFFING_ATTENDANCE_READ: (
        str(STAFFING_ATTENDANCE_EVIDENCE_ID),
        "staffing attendance observation",
    ),
    TOOL_COMPARISON_READ: (str(COMPARISON_EVIDENCE_ID), "comparison line observation"),
    TOOL_TELEMETRY_READ: (str(TELEMETRY_EVIDENCE_ID), "station telemetry observation"),
}


@dataclass(frozen=True, slots=True)
class EvidenceGatheringResult:
    evidence_nodes: tuple[dict[str, object], ...]
    tool_invocations: int
    stop_reason: ToolInvocationStopReason
    loop_iterations: int
    tool_execution_order: tuple[str, ...]
    planner_decisions: tuple[dict[str, object], ...]
    initial_evidence_ids: tuple[str, ...]


class _IncidentScopedToolInvoker:
    """Validates tool args against :class:`IncidentScope` before provider execution."""

    def __init__(
        self,
        *,
        inner: RuntimeToolInvoker,
        scope: IncidentScope,
        runtime_state: RuntimeState,
        investigation_phase: str,
    ) -> None:
        self._inner = inner
        self._scope = scope
        self._runtime_state = runtime_state
        self._investigation_phase = investigation_phase

    @property
    def registry(self) -> ToolRegistry:
        return self._inner.registry

    def invoke(self, state: RuntimeState, request: object, agent_id: str | None = None) -> object:
        from intergrax.tools.execution_models import ToolExecutionRequest

        if not isinstance(request, ToolExecutionRequest):
            raise TypeError("IncidentScopedToolInvoker expects ToolExecutionRequest")
        args = request.input.model_dump(mode="json")
        try:
            self._scope.validate_tool_input(request.tool_id, args)
        except IncidentScopeViolationError as exc:
            self._runtime_state.trace_event(
                component=TraceComponent.PLANNER,
                step="incident_scope_rejection",
                message="Incident scope rejected planner tool arguments",
                level=TraceLevel.WARNING,
                payload=IncidentScopeRejectionDiagV1(
                    tool_id=request.tool_id,
                    rejection_code=str(exc),
                    investigation_phase=self._investigation_phase,
                ),
            )
            raise
        return self._inner.invoke(state=state, request=request, agent_id=agent_id)


def build_catalog_tool_planner(
    *,
    runtime_state: RuntimeState,
    registry: ToolRegistry,
) -> CatalogToolPlanner:
    llm = runtime_state.context.config.llm_adapter
    if llm is None:
        raise RuntimeError("incident_planner_llm_missing")
    prompt_registry = runtime_state.context.prompt_registry
    catalog_path = runtime_state.context.config.prompt_catalog_path
    config = ToolPlanningConfig.default(
        registry=prompt_registry,
        catalog_path=catalog_path,
        investigation_prompt_id=INCIDENT_INVESTIGATION_POLICY_PROMPT_ID,
    )
    return CatalogToolPlanner(
        _service=ToolPlanningService(llm=llm, tools=registry, config=config)
    )


def build_investigation_planner_input(
    *,
    scope: IncidentScope,
    is_revision: bool,
    critic_feedback: Sequence[str] | None,
    gathered_evidence: Sequence[dict[str, object]],
) -> list[ChatMessage]:
    phase = "revision" if is_revision else "initial"
    scope_lines = [
        f"Investigation phase: {phase}",
        f"Permitted line_id: {scope.line_id}",
        f"Permitted station_id: {scope.station_id}",
        f"Permitted incident window: {scope.incident_window}",
        f"Permitted comparison window: {scope.comparison_window}",
        f"Permitted comparison line: {scope.comparison_line_id}",
        f"Permitted shift_id: {scope.shift_id}",
    ]
    if critic_feedback:
        scope_lines.append("Critic feedback requiring follow-up evidence:")
        scope_lines.extend(f"- {item}" for item in critic_feedback)
    if gathered_evidence:
        scope_lines.append("Already gathered evidence IDs:")
        scope_lines.extend(
            f"- {node.get('evidence_id')}" for node in gathered_evidence if node.get("evidence_id")
        )
    scope_lines.extend(
        [
            "Objective: gather distinguishing operational evidence for Line 4 target attainment degradation.",
            "Choose available tools autonomously; do not assume correlation equals causation.",
            "Stop tool gathering when sufficient evidence exists for the next reasoning phase.",
            "Do not request entities or windows outside the permitted scope.",
        ]
    )
    return [
        ChatMessage(role="system", content="\n".join(scope_lines)),
        ChatMessage(
            role="user",
            content=(
                "Investigate the incident by requesting production evidence tools within scope."
            ),
        ),
    ]


def _tool_call_id_to_evidence_id(
    tool_call_id: str,
    call_id_to_tool_name: dict[str, str],
) -> str | None:
    tool_name = call_id_to_tool_name.get(tool_call_id)
    if tool_name is None:
        return None
    mapped = _TOOL_EVIDENCE_MAP.get(tool_name)
    return mapped[0] if mapped else None


def _emit_planner_decision_traces(
    *,
    runtime_state: RuntimeState,
    investigation_phase: str,
    scope: IncidentScope,
    loop_result: object,
    call_id_to_tool_name: dict[str, str],
) -> tuple[dict[str, object], ...]:
    proof = getattr(loop_result, "investigation_proof", None)
    if proof is None or not proof.steps:
        return ()
    decisions: list[dict[str, object]] = []
    for step in proof.steps:
        selected_tools = tuple(
            call_id_to_tool_name[call_id]
            for call_id in step.next_tool_call_ids
            if call_id in call_id_to_tool_name
        )
        basis_evidence_ids = tuple(
            evidence_id
            for call_id in step.basis_tool_call_ids
            if (evidence_id := _tool_call_id_to_evidence_id(call_id, call_id_to_tool_name))
            is not None
        )
        payload = IncidentPlannerDecisionDiagV1(
            round_index=step.round_index,
            investigation_phase=investigation_phase,
            objective=step.public_reason or "gather incident evidence",
            selected_tool_ids=selected_tools,
            evidence_basis_tool_call_ids=step.basis_tool_call_ids,
            evidence_basis_evidence_ids=basis_evidence_ids,
            incident_line_id=scope.line_id,
            incident_station_id=scope.station_id,
        )
        runtime_state.trace_event(
            component=TraceComponent.PLANNER,
            step="incident_planner_decision",
            message="Incident investigator planner decision",
            level=TraceLevel.INFO,
            payload=payload,
        )
        decisions.append(payload.to_dict())
    return tuple(decisions)


def _extract_tool_outputs(
  loop_result: object,
) -> list[tuple[str, str, dict[str, object]]]:
    """Return (tool_call_id, tool_name, output_dict) from native tool messages."""
    outputs: list[tuple[str, str, dict[str, object]]] = []
    for message in getattr(loop_result, "appended_messages", []):
        if message.role != "tool" or not message.tool_call_id:
            continue
        tool_name = message.name or ""
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            outputs.append((message.tool_call_id, tool_name, payload))
    return outputs


def _evidence_nodes_from_tool_outputs(
    tool_outputs: Sequence[tuple[str, str, dict[str, object]]],
) -> tuple[dict[str, object], ...]:
    nodes: list[dict[str, object]] = []
    seen_evidence: set[str] = set()
    for _call_id, tool_name, payload in tool_outputs:
        mapped = _TOOL_EVIDENCE_MAP.get(tool_name)
        if mapped is None:
            continue
        evidence_id, label = mapped
        if evidence_id in seen_evidence:
            continue
        seen_evidence.add(evidence_id)
        nodes.append(
            {
                "evidence_id": evidence_id,
                "kind": "tool_result",
                "label": label,
                "payload": payload,
                "source_tool_id": tool_name,
            }
        )
    return tuple(nodes)


def gather_incident_evidence(
    *,
    runtime_state: RuntimeState,
    registry: ToolRegistry,
    scope: IncidentScope,
    is_revision: bool,
    critic_feedback: Sequence[str] | None = None,
    prior_evidence: Sequence[dict[str, object]] = (),
) -> EvidenceGatheringResult:
    investigation_phase = "revision" if is_revision else "initial"
    planner = build_catalog_tool_planner(runtime_state=runtime_state, registry=registry)
    inner_invoker = RuntimeToolInvoker(registry=registry, executor=RegistryToolExecutor(registry))
    invoker = _IncidentScopedToolInvoker(
        inner=inner_invoker,
        scope=scope,
        runtime_state=runtime_state,
        investigation_phase=investigation_phase,
    )
    planner_input = build_investigation_planner_input(
        scope=scope,
        is_revision=is_revision,
        critic_feedback=critic_feedback,
        gathered_evidence=prior_evidence,
    )
    try:
        loop_result = run_bounded_tool_loop(
            state=runtime_state,
            invoker=invoker,
            tool_planner=planner,
            planner_input=planner_input,
            allowed_tool_ids=SCENARIO_TOOL_IDS,
            max_iterations=MAX_INCIDENT_TOOL_LOOP_ITERATIONS,
            invocation_mode=ToolInvocationMode.BOUNDED_REACT,
        )
    except BudgetExceededError:
        raise RuntimeError("incident_evidence_gathering_budget_exceeded") from None

    if loop_result.stop_reason == "max_iterations":
        raise RuntimeError("incident_evidence_gathering_max_iterations")

    tool_outputs = _extract_tool_outputs(loop_result)
    call_id_to_tool_name = {call_id: tool_name for call_id, tool_name, _payload in tool_outputs}

    planner_decisions = _emit_planner_decision_traces(
        runtime_state=runtime_state,
        investigation_phase=investigation_phase,
        scope=scope,
        loop_result=loop_result,
        call_id_to_tool_name=call_id_to_tool_name,
    )
    tool_execution_order = tuple(tool_name for _call_id, tool_name, _payload in tool_outputs)
    runtime_state.trace_event(
        component=TraceComponent.PLANNER,
        step="incident_planner_stop",
        message="Incident evidence gathering stopped",
        level=TraceLevel.INFO,
        payload=IncidentPlannerStopDiagV1(
            investigation_phase=investigation_phase,
            stop_reason=loop_result.stop_reason,
            loop_iterations=loop_result.loop_iterations,
            tool_invocations=len(loop_result.tool_traces),
            selected_tool_order=tool_execution_order,
        ),
    )

    new_nodes = _evidence_nodes_from_tool_outputs(tool_outputs)
    merged: dict[str, dict[str, object]] = {
        str(node["evidence_id"]): dict(node) for node in prior_evidence if node.get("evidence_id")
    }
    for node in new_nodes:
        merged[str(node["evidence_id"])] = dict(node)
    evidence_nodes = tuple(merged.values())

    initial_ids = (
        str(WORKLOAD_EVIDENCE_ID),
        str(THROUGHPUT_EVIDENCE_ID),
        str(STAFFING_PRELIMINARY_EVIDENCE_ID),
    )
    return EvidenceGatheringResult(
        evidence_nodes=evidence_nodes,
        tool_invocations=len(loop_result.tool_traces),
        stop_reason=loop_result.stop_reason,
        loop_iterations=loop_result.loop_iterations,
        tool_execution_order=tool_execution_order,
        planner_decisions=planner_decisions,
        initial_evidence_ids=initial_ids,
    )
