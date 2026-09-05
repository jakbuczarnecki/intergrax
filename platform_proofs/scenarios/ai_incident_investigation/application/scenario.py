# © Artur Czarnecki. All rights reserved.

"""Incident investigation application entry — Nexus-backed execution via scenario runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    execute_scenario_task,
)
from intergrax.contracts.evidence_claims import EvidenceChallenge, EvidenceClaimSet, ClaimResolution
from intergrax.contracts.evidence_claims import validate_evidence_claim_id
from intergrax.runtime.diagnostics.investigation_contracts import (
    IncidentInvestigationInput,
    InvestigationConclusion,
    InvestigationConclusionStatus,
    validate_investigation_conclusion,
)
from intergrax.runtime.diagnostics import ProblemId
from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.runtime.decision_flow import DecisionFlowHostAction, DecisionFlowScope
from intergrax.runtime.decision_flow_host import (
    agent_execution_decision_context,
    agent_execution_identity_seed,
    build_agent_execution_flow_request,
    evaluate_agent_execution_flow,
)
from intergrax.runtime.migration.legacy_critic_contracts import LegacyCriticVerdict
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.application.critic_adapter import (
    apply_challenge_lifecycle,
    count_evaluator_loop_iterations_from_persisted_trace,
    first_failed_node_partial_verdict_from_persisted_trace,
    legacy_verdict_from_validation_errors,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_data_contracts import (
    IncidentOperationalData,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    INITIAL_CLAIM_ID,
    IncidentInvestigatorAgent,
    INVESTIGATOR_AGENT_ID,
    INVESTIGATOR_CAPABILITY,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    claim_id_for_hypothesis,
    parse_claim_hypothesis_bindings,
)
from platform_proofs.scenarios.ai_incident_investigation.application.tools import (
    ScenarioEvidenceStore,
    register_scenario_tools,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    INVESTIGATOR_NODE_ID,
    ScenarioRuntimeComposition,
    build_scenario_environment_profile,
    build_scenario_runtime_composition,
    prepare_incident_execution_runtime,
    trace_reader_from_composition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    COMPLETION_SUPPORTED_DIAGNOSIS,
    COMPLETION_UNRESOLVED,
)
from platform_proofs.scenarios.ai_incident_investigation.application.execution_payload import (
    domain_payload_from_execution,
)
from platform_proofs.scenarios.ai_incident_investigation.application.incident_scope import IncidentScope
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    IncidentInvestigationValidationEngine,
    UNSUPPORTED_INFERENCE_ERROR,
    apply_critic_claim_resolutions,
)

OUTCOME_RESOLVED = "RESOLVED"
OUTCOME_UNRESOLVED = "UNRESOLVED"
STANDALONE_SCENARIO_TENANT_ID = "scenario-tenant"
SYNTHETIC_SCENARIO_TENANT_ID = STANDALONE_SCENARIO_TENANT_ID
TERMINAL_STATE_NOT_ACCEPTED = "incident_terminal_state_not_accepted"
EVALUATOR_LOOP_MAX_ITERATIONS = 2


def is_resolved_completion(
    *,
    critic_verdict_passed: bool,
    has_supported_diagnosis: bool,
    completion_mode: str,
) -> bool:
    return (
        critic_verdict_passed
        and has_supported_diagnosis
        and completion_mode == COMPLETION_SUPPORTED_DIAGNOSIS
    )


def is_epistemic_unresolved_completion(
    *,
    critic_verdict_passed: bool,
    has_supported_diagnosis: bool,
    completion_mode: str,
) -> bool:
    return (
        critic_verdict_passed
        and not has_supported_diagnosis
        and completion_mode == COMPLETION_UNRESOLVED
    )


def derive_terminal_outcome(
    *,
    critic_verdict_passed: bool,
    has_supported_diagnosis: bool,
    completion_mode: str,
) -> str:
    if is_resolved_completion(
        critic_verdict_passed=critic_verdict_passed,
        has_supported_diagnosis=has_supported_diagnosis,
        completion_mode=completion_mode,
    ):
        return OUTCOME_RESOLVED
    if is_epistemic_unresolved_completion(
        critic_verdict_passed=critic_verdict_passed,
        has_supported_diagnosis=has_supported_diagnosis,
        completion_mode=completion_mode,
    ):
        return OUTCOME_UNRESOLVED
    raise RuntimeError(TERMINAL_STATE_NOT_ACCEPTED)


@dataclass(frozen=True, slots=True)
class ScenarioExecutionResult:
    outcome: str
    terminal_summary: str
    claim_set: dict[str, Any]
    evidence_nodes: tuple[dict[str, Any], ...]
    initial_evidence_nodes: tuple[dict[str, Any], ...]
    tool_trace_count: int
    tool_invocations: int
    evaluator_loop_iterations: int
    critic_challenged: bool
    revision_used_tools: bool
    revision_pass: bool
    critic_verdict_passed: bool
    leak_scan_blob: str
    failed_critic_verdict: LegacyCriticVerdict | None
    evidence_challenge: EvidenceChallenge | None
    claim_hypothesis_bindings: tuple[dict[str, Any], ...] = ()
    challenged_claim_id: str | None = None
    planner_decisions: tuple[dict[str, Any], ...] = ()
    tool_execution_order: tuple[str, ...] = ()
    evidence_gathering_stop_reason: str = ""
    investigation_conclusion: InvestigationConclusion | None = None
    investigated_problem_ids: tuple[ProblemId, ...] = ()
    execution_tenant_id: str = STANDALONE_SCENARIO_TENANT_ID


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeBundle:
    operational_data: IncidentOperationalData
    registry: ToolRegistry
    investigator: IncidentInvestigatorAgent
    runtime_composition: ScenarioRuntimeComposition
    evidence_store: ScenarioEvidenceStore
    investigation_input: IncidentInvestigationInput | None = None


def build_runtime_bundle(
    *,
    operational_data: IncidentOperationalData,
    runtime_composition: ScenarioRuntimeComposition | None = None,
    tenant_id: str = STANDALONE_SCENARIO_TENANT_ID,
    investigation_input: IncidentInvestigationInput | None = None,
    llm_adapter_override: LLMAdapter | None = None,
) -> ScenarioRuntimeBundle:
    environment = (
        runtime_composition.environment
        if runtime_composition is not None
        else build_scenario_environment_profile()
    )
    tool_registry = (
        runtime_composition.tool_registry
        if runtime_composition is not None
        else ToolRegistry()
    )
    evidence_store = register_scenario_tools(tool_registry, operational_data)
    composition = runtime_composition or ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=tool_registry,
        llm_adapter_override=llm_adapter_override,
    )
    if not composition.is_platform_attached:
        build_scenario_runtime_composition(
            registry=tool_registry,
            tenant_id=tenant_id,
            environment=composition.environment,
            composition=composition,
        )
    composition.tool_registry = composition.platform.env_wiring.tool_wiring.registry
    investigator = IncidentInvestigatorAgent(
        registry=tool_registry,
        station_id=operational_data.station_id,
        runtime_composition=composition,
        incident_scope=IncidentScope.from_operational_defaults(
            station_id=operational_data.station_id,
        ),
        evidence_store=evidence_store,
        investigation_input=investigation_input,
    )
    if not composition.platform.registry.has(INVESTIGATOR_AGENT_ID):
        composition.platform.registry.register(investigator)
    return ScenarioRuntimeBundle(
        operational_data=operational_data,
        registry=tool_registry,
        investigator=investigator,
        runtime_composition=composition,
        evidence_store=evidence_store,
        investigation_input=investigation_input,
    )


def investigation_conclusion_status_from_outcome(outcome: str) -> InvestigationConclusionStatus:
    if outcome == OUTCOME_RESOLVED:
        return InvestigationConclusionStatus.SUPPORTED
    if outcome == OUTCOME_UNRESOLVED:
        return InvestigationConclusionStatus.UNRESOLVED
    return InvestigationConclusionStatus.NOT_ACCEPTED


def build_investigation_conclusion(
    *,
    outcome: str,
    investigated_problem_ids: tuple[ProblemId, ...],
    claim_set: EvidenceClaimSet | None = None,
    summary: str | None = None,
) -> InvestigationConclusion | None:
    if not investigated_problem_ids:
        return None
    return validate_investigation_conclusion(
        InvestigationConclusion(
            status=investigation_conclusion_status_from_outcome(outcome),
            investigated_problem_ids=investigated_problem_ids,
            claim_set=claim_set,
            summary=summary or None,
        )
    )


def _leak_scan_blob(
    claim_set: dict[str, Any],
    evidence_nodes: tuple[dict[str, Any], ...],
) -> str:
    return json.dumps({"claim_set": claim_set, "evidence_nodes": evidence_nodes})


def _persisted_trace_events(
    composition: ScenarioRuntimeComposition,
    run_id: str,
    tenant_id: str,
) -> list[dict[str, object]]:
    reader = trace_reader_from_composition(composition)
    if reader is None:
        return []
    persisted = reader.read_run(run_id, tenant_id)
    return [dict(item) for item in persisted.events if isinstance(item, dict)]


async def execute_resolved_skeleton(
    bundle: ScenarioRuntimeBundle,
    *,
    semantic_verification_enabled: bool = False,
    validation_engine: IncidentInvestigationValidationEngine | None = None,
    evaluator_loop_max_iterations: int = EVALUATOR_LOOP_MAX_ITERATIONS,
    max_decision_revisions: int = 0,
) -> ScenarioExecutionResult:
    composition = bundle.runtime_composition
    validation_engine = prepare_incident_execution_runtime(
        composition,
        validation_engine=validation_engine,
        semantic_verification_enabled=semantic_verification_enabled,
        evaluator_loop_max_iterations=evaluator_loop_max_iterations,
        max_decision_revisions=max_decision_revisions,
    )
    platform = composition.platform

    execution_tenant_id = (
        bundle.investigation_input.tenant_id
        if bundle.investigation_input is not None
        else STANDALONE_SCENARIO_TENANT_ID
    )
    investigated_problem_ids: tuple[ProblemId, ...] = ()
    if bundle.investigation_input is not None:
        investigated_problem_ids = tuple(
            context.problem.problem_id
            for context in bundle.investigation_input.problem_contexts
        )

    platform_result = await execute_scenario_task(
        platform,
        ScenarioExecutionRequest(
            tenant_id=execution_tenant_id,
            message="Investigate Line 4 target attainment degradation",
            capability=INVESTIGATOR_CAPABILITY,
        ),
    )
    task_result = platform_result.task_result
    run_id = str(platform_result.run_id)
    final_execution = task_result.execution_result
    if final_execution is None:
        if task_result.state.value == "failed":
            raise RuntimeError(TERMINAL_STATE_NOT_ACCEPTED)
        raise RuntimeError("no agent executions produced")

    trace_events = _persisted_trace_events(composition, run_id, execution_tenant_id)
    domain_payload = domain_payload_from_execution(final_execution)
    evaluator_loop_iterations = count_evaluator_loop_iterations_from_persisted_trace(
        trace_events,
        node_id=INVESTIGATOR_NODE_ID,
    )
    failed_critic_verdict = first_failed_node_partial_verdict_from_persisted_trace(
        trace_events,
        node_id=INVESTIGATOR_NODE_ID,
    )
    revision_pass = bool(domain_payload.get("revision_pass", False))
    if failed_critic_verdict is None and revision_pass:
        failed_critic_verdict = legacy_verdict_from_validation_errors(
            [UNSUPPORTED_INFERENCE_ERROR],
            node_id=INVESTIGATOR_NODE_ID,
        )
    bindings = parse_claim_hypothesis_bindings(domain_payload.get("claim_hypothesis_bindings"))
    resolved_claim_set = apply_critic_claim_resolutions(
        EvidenceClaimSet.model_validate(dict(domain_payload.get("claim_set", {}))),
        domain_payload,
        bindings=bindings,
    )
    claim_set = resolved_claim_set.model_dump(mode="json")
    evidence_nodes = tuple(domain_payload.get("evidence_nodes", []))
    initial_ids_raw = domain_payload.get("initial_evidence_ids", [])
    if isinstance(initial_ids_raw, list) and initial_ids_raw:
        initial_id_set = {str(item) for item in initial_ids_raw}
        initial_evidence_nodes = tuple(
            node for node in evidence_nodes if str(node.get("evidence_id")) in initial_id_set
        )
    else:
        initial_evidence_nodes = evidence_nodes
    tool_invocations = int(domain_payload.get("tool_invocations", 0))
    planner_decisions_raw = domain_payload.get("planner_decisions", [])
    planner_decisions = tuple(
        dict(item) for item in planner_decisions_raw if isinstance(item, dict)
    )
    tool_order_raw = domain_payload.get("tool_execution_order", [])
    tool_execution_order = tuple(str(item) for item in tool_order_raw if item)
    evidence_gathering_stop_reason = str(domain_payload.get("evidence_gathering_stop_reason", ""))

    critic_challenged = failed_critic_verdict is not None and not failed_critic_verdict.passed

    decision_gate = composition.platform.nexus_loop.peek_decision_flow_gate()
    if decision_gate is not None and decision_gate.supports_scope(DecisionFlowScope.GRAPH_FINAL):
        active_run_id, active_attempt_id = require_active_execution_identity()
        decision_context = agent_execution_decision_context(
            task_id=task_result.task_id,
            run_id=active_run_id,
            attempt_id=active_attempt_id,
            tenant_id=execution_tenant_id,
        )
        identity_seed = agent_execution_identity_seed(
            context=decision_context,
            namespace="graph.final",
            subject=INVESTIGATOR_NODE_ID,
        )
        flow_request = build_agent_execution_flow_request(
            execution=final_execution,
            identity_seed=identity_seed,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        )
        flow_result = await evaluate_agent_execution_flow(decision_gate, flow_request)
        critic_verdict_passed = flow_result.host_action is DecisionFlowHostAction.CONTINUE
    else:
        final_validation = validation_engine.validate(
            final_execution,
            contract=bundle.investigator.get_contract(),
            capability=INVESTIGATOR_CAPABILITY,
        )
        critic_verdict_passed = final_validation.valid

    evidence_challenge: EvidenceChallenge | None = None
    claim_set_model = resolved_claim_set
    has_supported_diagnosis = any(
        claim.resolution is ClaimResolution.SUPPORTED for claim in claim_set_model.claims
    )
    completion_mode = str(domain_payload.get("completion_mode", COMPLETION_SUPPORTED_DIAGNOSIS))

    first_bindings = bindings
    if critic_challenged and failed_critic_verdict is not None:
        challenged_claim_id = claim_id_for_hypothesis(first_bindings, "H1")
        claim_set, evidence_challenge = apply_challenge_lifecycle(
            claim_set,
            failed_critic_verdict,
            claim_id=validate_evidence_claim_id(challenged_claim_id or str(INITIAL_CLAIM_ID)),
            initial_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
            resolving_evidence_ids=(
                TELEMETRY_EVIDENCE_ID,
                COMPARISON_EVIDENCE_ID,
                STAFFING_ATTENDANCE_EVIDENCE_ID,
            ),
            resolved=critic_verdict_passed and has_supported_diagnosis,
            satisfied_description=(
                "Follow-up comparison, attendance, and telemetry gathered via platform tools"
            ),
        )

    if task_result.state.value != "completed" and critic_verdict_passed:
        raise RuntimeError(f"investigator task not completed: {task_result.state}")

    outcome = derive_terminal_outcome(
        critic_verdict_passed=critic_verdict_passed,
        has_supported_diagnosis=has_supported_diagnosis,
        completion_mode=completion_mode,
    )
    leak_blob = _leak_scan_blob(claim_set, evidence_nodes)
    investigation_conclusion = build_investigation_conclusion(
        outcome=outcome,
        investigated_problem_ids=investigated_problem_ids,
        claim_set=claim_set_model,
        summary=final_execution.summary or None,
    )

    return ScenarioExecutionResult(
        outcome=outcome,
        terminal_summary=final_execution.summary or "",
        claim_set=claim_set,
        evidence_nodes=evidence_nodes,
        initial_evidence_nodes=initial_evidence_nodes,
        tool_trace_count=tool_invocations,
        tool_invocations=tool_invocations,
        evaluator_loop_iterations=evaluator_loop_iterations,
        critic_challenged=critic_challenged,
        revision_used_tools=revision_pass and tool_invocations >= 6,
        revision_pass=revision_pass,
        critic_verdict_passed=critic_verdict_passed,
        leak_scan_blob=leak_blob,
        failed_critic_verdict=failed_critic_verdict,
        evidence_challenge=evidence_challenge,
        claim_hypothesis_bindings=tuple(
            binding.model_dump(mode="json") for binding in bindings
        ),
        challenged_claim_id=claim_id_for_hypothesis(first_bindings, "H1"),
        planner_decisions=planner_decisions,
        tool_execution_order=tool_execution_order,
        evidence_gathering_stop_reason=evidence_gathering_stop_reason,
        investigation_conclusion=investigation_conclusion,
        investigated_problem_ids=investigated_problem_ids,
        execution_tenant_id=execution_tenant_id,
    )


async def execute_with_completion_gate_blocked(bundle: ScenarioRuntimeBundle) -> ScenarioExecutionResult:
    """Real Nexus path where Decision revision budget is exhausted within platform loop."""
    return await execute_resolved_skeleton(
        bundle,
        semantic_verification_enabled=True,
        evaluator_loop_max_iterations=1,
        max_decision_revisions=0,
    )
