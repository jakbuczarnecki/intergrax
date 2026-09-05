# © Artur Czarnecki. All rights reserved.

"""Autonomous incident evidence gathering via canonical bounded tool loop (APP-2A)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationStopReason
from intergrax.runtime.nexus.tools.tool_invoker_protocol import ToolInvokerProtocol
from intergrax.runtime.nexus.tools.tool_loop import run_bounded_tool_loop
from intergrax.runtime.nexus.tools.tool_planning_config import ToolPlanningConfig
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.incident_scope import (
    IncidentScope,
    IncidentScopeViolationError,
)
from platform_proofs.scenarios.ai_incident_investigation.application.observability import (
    IncidentPlannerDecisionDiagV1,
    IncidentPlannerStopDiagV1,
    IncidentScopeRejectionDiagV1,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPARISON_EVIDENCE_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    ANALYSIS_TOOL_IDS,
    RAW_EVIDENCE_TOOL_IDS,
    SCENARIO_TOOL_IDS,
    ScenarioEvidenceStore,
    TOOL_COMPARISON_EVALUATE,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_STAFFING_EVALUATE,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_TELEMETRY_EVALUATE,
    TOOL_TELEMETRY_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_WORKLOAD_EVALUATE,
    TOOL_WORKLOAD_READ,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    H3_FORGED_WITHOUT_TELEMETRY_ERROR,
    MISSING_COMPARISON_ERROR,
    UNSUPPORTED_INFERENCE_ERROR,
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

_ANALYSIS_TOOL_LABELS: dict[str, str] = {
    TOOL_WORKLOAD_EVALUATE: "workload-throughput analysis",
    TOOL_STAFFING_EVALUATE: "staffing consistency analysis",
    TOOL_COMPARISON_EVALUATE: "peer-line comparison analysis",
    TOOL_TELEMETRY_EVALUATE: "station telemetry analysis",
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
        inner: ToolInvokerProtocol,
        registry: ToolRegistry,
        scope: IncidentScope,
        runtime_state: RuntimeState,
        investigation_phase: str,
        evidence_store: ScenarioEvidenceStore | None = None,
    ) -> None:
        self._inner = inner
        self._registry = registry
        self._scope = scope
        self._runtime_state = runtime_state
        self._investigation_phase = investigation_phase
        self._evidence_store = evidence_store

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

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
            return ToolExecutionResult.fail("incident_scope_violation", str(exc))
        result = self._inner.invoke(state=state, request=request, agent_id=agent_id)
        if self._evidence_store is not None and isinstance(result, ToolExecutionResult):
            if result.success and result.output is not None:
                payload = (
                    result.output.model_dump(mode="json")
                    if hasattr(result.output, "model_dump")
                    else dict(result.output)
                    if isinstance(result.output, dict)
                    else None
                )
                if payload is not None:
                    mapped = _TOOL_EVIDENCE_MAP.get(request.tool_id)
                    if mapped is not None:
                        self._evidence_store.record(
                            mapped[0],
                            payload,
                            source_tool_id=request.tool_id,
                        )
        return result


def _analysis_evidence_id(tool_name: str, payload: dict[str, object]) -> str:
    analysis_type = str(payload.get("analysis_type", tool_name.split(".")[-1]))
    source_ids = payload.get("source_evidence_ids")
    if isinstance(source_ids, list) and source_ids:
        suffix = "_".join(str(item).replace(".", "_") for item in source_ids)
    else:
        suffix = tool_name.replace(".", "_")
    return f"evidence.analysis.{analysis_type}.{suffix}"


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
        f"Permitted incident reference line: {scope.reference_line_id}",
        f"Permitted comparison peer line: {scope.comparison_line_id}",
        (
            "Window labels: use incident_window for workload, throughput, staffing, and telemetry; "
            f"use {scope.comparison_window} for production.comparison.read only."
        ),
        (
            "For production.comparison.read use reference_line_id="
            f"{scope.reference_line_id}, comparison_line_id={scope.comparison_line_id}, "
            f"and window={scope.comparison_window}."
        ),
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
            "Raw evidence acquisition tools and deterministic domain analysis tools are available.",
            "Use analysis tools when bounded deterministic comparison improves confidence.",
            "Choose available scenario tools autonomously based on current information gaps.",
            "Do not assume correlation equals causation.",
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
        if message.role != "tool":
            continue
        tool_name = message.name or ""
        if not tool_name:
            continue
        tool_call_id = message.tool_call_id
        if not tool_call_id:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            outputs.append((tool_call_id, tool_name, payload))
    return outputs


def _evidence_nodes_from_tool_outputs(
    tool_outputs: Sequence[tuple[str, str, dict[str, object]]],
) -> tuple[dict[str, object], ...]:
    nodes: list[dict[str, object]] = []
    seen_evidence: set[str] = set()
    for _call_id, tool_name, payload in tool_outputs:
        mapped = _TOOL_EVIDENCE_MAP.get(tool_name)
        if mapped is not None:
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
            continue
        if tool_name in _ANALYSIS_TOOL_LABELS:
            evidence_id = _analysis_evidence_id(tool_name, payload)
            if evidence_id in seen_evidence:
                continue
            seen_evidence.add(evidence_id)
            source_ids = payload.get("source_evidence_ids", [])
            nodes.append(
                {
                    "evidence_id": evidence_id,
                    "kind": "derived_analysis",
                    "label": _ANALYSIS_TOOL_LABELS[tool_name],
                    "payload": payload,
                    "source_tool_id": tool_name,
                    "source_evidence_ids": list(source_ids)
                    if isinstance(source_ids, list)
                    else [],
                }
            )
    return tuple(nodes)


def _scoped_tool_args(tool_id: str, scope: IncidentScope) -> dict[str, str]:
    if tool_id in {TOOL_WORKLOAD_READ, TOOL_THROUGHPUT_READ}:
        return {"line_id": scope.line_id, "window": scope.incident_window}
    if tool_id in {TOOL_STAFFING_SCHEDULE_READ, TOOL_STAFFING_ATTENDANCE_READ}:
        return {
            "line_id": scope.line_id,
            "shift_id": scope.shift_id,
            "window": scope.incident_window,
        }
    if tool_id == TOOL_COMPARISON_READ:
        return {
            "reference_line_id": scope.reference_line_id,
            "comparison_line_id": scope.comparison_line_id,
            "window": scope.comparison_window,
        }
    if tool_id == TOOL_TELEMETRY_READ:
        return {"station_id": scope.station_id, "window": scope.incident_window}
    raise ValueError(f"unsupported supplemental tool id: {tool_id}")


_REVISION_EVIDENCE_TOOLS: tuple[str, ...] = (
    TOOL_WORKLOAD_READ,
    TOOL_THROUGHPUT_READ,
    TOOL_STAFFING_SCHEDULE_READ,
    TOOL_COMPARISON_READ,
    TOOL_STAFFING_ATTENDANCE_READ,
    TOOL_TELEMETRY_READ,
)


def _tools_for_critic_feedback(critic_feedback: Sequence[str] | None) -> tuple[str, ...]:
    if not critic_feedback:
        return ()
    joined = " ".join(str(item) for item in critic_feedback)
    required: list[str] = []
    if MISSING_COMPARISON_ERROR in joined:
        required.append(TOOL_COMPARISON_READ)
    if H3_FORGED_WITHOUT_TELEMETRY_ERROR in joined:
        required.append(TOOL_TELEMETRY_READ)
    if UNSUPPORTED_INFERENCE_ERROR in joined:
        required.extend((TOOL_COMPARISON_READ, TOOL_TELEMETRY_READ))
    if "missing_distinguishing_equipment_evidence" in joined:
        required.extend((TOOL_COMPARISON_READ, TOOL_TELEMETRY_READ))
    if "staffing attendance" in joined.lower() or "attendance confirmation" in joined.lower():
        required.append(TOOL_STAFFING_ATTENDANCE_READ)
    deduped: list[str] = []
    for tool_id in required:
        if tool_id not in deduped:
            deduped.append(tool_id)
    return tuple(deduped)


def _revision_supplement_tool_ids(
    *,
    is_revision: bool,
    critic_feedback: Sequence[str] | None,
) -> tuple[str, ...]:
    if is_revision:
        return _REVISION_EVIDENCE_TOOLS
    return _tools_for_critic_feedback(critic_feedback)


def _invoke_supplemental_tools(
    *,
    tool_ids: Sequence[str],
    invoker: _IncidentScopedToolInvoker,
    runtime_state: RuntimeState,
    scope: IncidentScope,
    existing_evidence_ids: set[str],
) -> list[tuple[str, str, dict[str, object]]]:
    outputs: list[tuple[str, str, dict[str, object]]] = []
    for index, tool_id in enumerate(tool_ids):
        mapped = _TOOL_EVIDENCE_MAP.get(tool_id)
        if mapped is not None and mapped[0] in existing_evidence_ids:
            continue
        tool = invoker.registry.get(tool_id)
        input_model = tool.contract.input_schema
        request = ToolExecutionRequest(
            run_id=runtime_state.run_id,
            step_id=f"supplement_{index}",
            tool_id=tool_id,
            input=input_model.model_validate(_scoped_tool_args(tool_id, scope)),
            idempotency_key=f"incident_supplement:{tool_id}:{index}",
        )
        result = invoker.invoke(runtime_state, request)
        if not isinstance(result, ToolExecutionResult) or not result.success:
            continue
        payload = (
            result.output.model_dump(mode="json")
            if hasattr(result.output, "model_dump")
            else dict(result.output)
            if isinstance(result.output, dict)
            else None
        )
        if payload is None:
            continue
        call_id = f"supplement_{tool_id.replace('.', '_')}_{index}"
        outputs.append((call_id, tool_id, payload))
        if mapped is not None:
            existing_evidence_ids.add(mapped[0])
    return outputs


def gather_incident_evidence(
    *,
    runtime_state: RuntimeState,
    registry: ToolRegistry,
    scope: IncidentScope,
    is_revision: bool,
    critic_feedback: Sequence[str] | None = None,
    prior_evidence: Sequence[dict[str, object]] = (),
    evidence_store: ScenarioEvidenceStore | None = None,
) -> EvidenceGatheringResult:
    if evidence_store is not None:
        for node in prior_evidence:
            evidence_id = node.get("evidence_id")
            payload = node.get("payload")
            source_tool_id = node.get("source_tool_id", "prior_evidence")
            if evidence_id and isinstance(payload, dict):
                evidence_store.record(
                    str(evidence_id),
                    payload,
                    source_tool_id=str(source_tool_id),
                )

    investigation_phase = "revision" if is_revision else "initial"
    canonical_invoker = runtime_state.context.config.tool_invoker
    if canonical_invoker is None:
        raise RuntimeError("incident_runtime_tool_invoker_missing")
    invoker = _IncidentScopedToolInvoker(
        inner=canonical_invoker,
        registry=registry,
        scope=scope,
        runtime_state=runtime_state,
        investigation_phase=investigation_phase,
        evidence_store=evidence_store,
    )

    allowed_tool_ids = tuple(SCENARIO_TOOL_IDS)
    planner = build_catalog_tool_planner(runtime_state=runtime_state, registry=registry)
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
            allowed_tool_ids=allowed_tool_ids,
            max_iterations=MAX_INCIDENT_TOOL_LOOP_ITERATIONS,
            invocation_mode=ToolInvocationMode.BOUNDED_REACT,
        )
    except BudgetExceededError:
        raise RuntimeError("incident_evidence_gathering_budget_exceeded") from None

    if loop_result.stop_reason == "max_iterations":
        raise RuntimeError("incident_evidence_gathering_max_iterations")

    tool_outputs = _extract_tool_outputs(loop_result)
    supplemental_outputs = _invoke_supplemental_tools(
        tool_ids=_revision_supplement_tool_ids(
            is_revision=is_revision,
            critic_feedback=critic_feedback,
        ),
        invoker=invoker,
        runtime_state=runtime_state,
        scope=scope,
        existing_evidence_ids={
            str(node["evidence_id"])
            for node in prior_evidence
            if node.get("evidence_id")
        }
        | {
            mapped[0]
            for _call_id, tool_name, _payload in tool_outputs
            if (mapped := _TOOL_EVIDENCE_MAP.get(tool_name)) is not None
        },
    )
    tool_outputs = list(tool_outputs) + supplemental_outputs
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
            tool_invocations=len(loop_result.tool_traces) + len(supplemental_outputs),
            selected_tool_order=tool_execution_order,
        ),
    )

    merged: dict[str, dict[str, object]] = {
        str(node["evidence_id"]): dict(node) for node in prior_evidence if node.get("evidence_id")
    }
    for node in _evidence_nodes_from_tool_outputs(tool_outputs):
        merged[str(node["evidence_id"])] = dict(node)
    if evidence_store is not None:
        for node in evidence_store.evidence_nodes():
            evidence_id = node.get("evidence_id")
            if evidence_id:
                merged[str(evidence_id)] = dict(node)
    evidence_nodes = tuple(merged.values())

    initial_ids = (
        str(WORKLOAD_EVIDENCE_ID),
        str(THROUGHPUT_EVIDENCE_ID),
        str(STAFFING_PRELIMINARY_EVIDENCE_ID),
    )
    return EvidenceGatheringResult(
        evidence_nodes=evidence_nodes,
        tool_invocations=len(loop_result.tool_traces) + len(supplemental_outputs),
        stop_reason=loop_result.stop_reason,
        loop_iterations=loop_result.loop_iterations,
        tool_execution_order=tool_execution_order,
        planner_decisions=planner_decisions,
        initial_evidence_ids=initial_ids,
    )
