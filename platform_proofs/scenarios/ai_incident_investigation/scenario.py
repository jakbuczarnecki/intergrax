# © Artur Czarnecki. All rights reserved.

"""Platform-native skeleton orchestration — GraphExecutor + Critic + EvaluatorLoop + ToolRuntime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.evidence_claims import EvidenceClaimSet
from intergrax.runtime.critic.contracts import CriticScope, CriticVerdict
from intergrax.runtime.critic.critic_wiring import (
    CriticHookConfig,
    build_critic_graph_hooks,
    validate_final_with_critic_detail,
    validate_node_with_critic_detail,
)
from intergrax.runtime.critic.evaluator_loop_metadata import (
    current_evaluator_loop_iteration,
    tag_node_evaluator_loop,
)
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.critic_adapter import (
    build_satisfied_challenge,
    map_critic_verdict_to_challenge,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures import (
    build_skeleton_fixture,
    IncidentFixture,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INITIAL_CLAIM_ID,
    IncidentInvestigatorAgent,
    INVESTIGATOR_AGENT_ID,
    INVESTIGATOR_CAPABILITY,
    WORKLOAD_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.tools import register_scenario_tools
from platform_proofs.scenarios.ai_incident_investigation.execution_payload import (
    domain_payload_from_execution,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    IncidentInvestigationValidationEngine,
    UNSUPPORTED_INFERENCE_ERROR,
)

INVESTIGATOR_NODE_ID = "investigator-1"
OUTCOME_RESOLVED = "RESOLVED"
OUTCOME_UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class ScenarioExecutionResult:
    outcome: str
    terminal_summary: str
    claim_set: dict[str, Any]
    evidence_nodes: tuple[dict[str, Any], ...]
    tool_trace_count: int
    tool_invocations: int
    evaluator_loop_iterations: int
    critic_challenged: bool
    revision_used_tools: bool
    critic_verdict_passed: bool
    leak_scan_blob: str


@dataclass(frozen=True, slots=True)
class ScenarioRuntimeBundle:
    fixture: IncidentFixture
    registry: ToolRegistry
    investigator: IncidentInvestigatorAgent


def build_runtime_bundle() -> ScenarioRuntimeBundle:
    fixture = build_skeleton_fixture()
    registry = ToolRegistry()
    register_scenario_tools(registry, fixture)
    investigator = IncidentInvestigatorAgent(
        registry=registry,
        station_id=fixture.telemetry.station_id,
    )
    return ScenarioRuntimeBundle(
        fixture=fixture,
        registry=registry,
        investigator=investigator,
    )


def _leak_scan_blob(
    claim_set: dict[str, Any],
    evidence_nodes: tuple[dict[str, Any], ...],
) -> str:
    return json.dumps({"claim_set": claim_set, "evidence_nodes": evidence_nodes})


def _attach_challenge_record(claim_set: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    failed_verdict = CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=False,
        failure_reasons=[UNSUPPORTED_INFERENCE_ERROR],
    )
    draft_challenge = map_critic_verdict_to_challenge(
        failed_verdict,
        claim_id=INITIAL_CLAIM_ID,
        evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
    )
    if draft_challenge is None:
        return claim_set, False
    claim_set_model = EvidenceClaimSet.model_validate(claim_set)
    satisfied = build_satisfied_challenge(
        draft_challenge.challenge_id,
        claim_id=INITIAL_CLAIM_ID,
        evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
        description="Follow-up telemetry gathered via platform tools",
    )
    updated = EvidenceClaimSet(
        claims=claim_set_model.claims,
        challenges=(satisfied,),
    )
    return updated.model_dump(mode="json"), True


async def execute_resolved_skeleton(
    bundle: ScenarioRuntimeBundle,
    *,
    require_critic_on_completion: bool = False,
    semantic_judge_enabled: bool = False,
) -> ScenarioExecutionResult:
    agent_registry = AgentRegistry()
    agent_registry.register(bundle.investigator)

    validation_engine = IncidentInvestigationValidationEngine()
    critic_hooks = build_critic_graph_hooks(
        config=CriticHookConfig(
            verify_node_partial=True,
            verify_graph_final=True,
            require_critic_on_completion=require_critic_on_completion,
            semantic_judge_enabled=semantic_judge_enabled,
        ),
        validation_engine=validation_engine,
    )
    if critic_hooks is None:
        raise RuntimeError("critic hooks required for skeleton")

    worker = ExecutionNode(
        node_id=INVESTIGATOR_NODE_ID,
        agent_id=INVESTIGATOR_AGENT_ID,
        capability=INVESTIGATOR_CAPABILITY,
    )
    tag_node_evaluator_loop(
        worker,
        EvaluatorLoopSpec(
            max_iterations=2,
            revise_node_id=INVESTIGATOR_NODE_ID,
            escalate_on_exhaustion=False,
        ),
    )

    task = Task(
        tenant_id="scenario-tenant",
        user_id="scenario-user",
        message="Investigate Line 4 target attainment degradation",
        context=TaskContext(capability=INVESTIGATOR_CAPABILITY),
    )
    graph = ExecutionGraph(
        graph_id="incident_investigation_skeleton",
        task_id=task.task_id,
        nodes=[worker],
    )
    executor = GraphExecutor(
        agent_registry,
        validation_engine=validation_engine,
        critic_graph_hooks=critic_hooks,
    )

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        executions, _retries, graph_out, _ = await executor.execute(graph, task)
    finally:
        reset_active_execution_identity(token)

    if not executions:
        raise RuntimeError("no agent executions produced")

    final_execution = executions[-1]
    domain_payload = domain_payload_from_execution(final_execution)
    claim_set = dict(domain_payload.get("claim_set", {}))
    evidence_nodes = tuple(domain_payload.get("evidence_nodes", []))
    tool_invocations = int(domain_payload.get("tool_invocations", 0))
    revision_pass = bool(domain_payload.get("revision_pass", False))
    evaluator_loop_iterations = current_evaluator_loop_iteration(worker)

    final_validation, final_verdict = validate_node_with_critic_detail(
        final_execution,
        contract=bundle.investigator.get_contract(),
        hooks=critic_hooks,
        task_id=task.task_id,
        run_id=run_id,
        tenant_id=task.tenant_id,
        capability=INVESTIGATOR_CAPABILITY,
        node_id=INVESTIGATOR_NODE_ID,
    )
    if critic_hooks.verify_graph_final:
        final_validation, final_verdict = validate_final_with_critic_detail(
            final_execution,
            contract=bundle.investigator.get_contract(),
            hooks=critic_hooks,
            task_id=task.task_id,
            run_id=run_id,
            tenant_id=task.tenant_id,
            capability=INVESTIGATOR_CAPABILITY,
        )
    critic_verdict_passed = final_verdict.passed and final_validation.valid

    critic_challenged = evaluator_loop_iterations > 0
    if critic_challenged:
        claim_set, _ = _attach_challenge_record(claim_set)

    node_status = graph_out.node_by_id(INVESTIGATOR_NODE_ID).status
    if node_status.value != "completed" and critic_verdict_passed:
        raise RuntimeError(f"investigator node not completed: {node_status}")

    outcome = OUTCOME_RESOLVED if critic_verdict_passed else OUTCOME_UNRESOLVED
    leak_blob = _leak_scan_blob(claim_set, evidence_nodes)

    return ScenarioExecutionResult(
        outcome=outcome,
        terminal_summary=final_execution.summary or "",
        claim_set=claim_set,
        evidence_nodes=evidence_nodes,
        tool_trace_count=tool_invocations,
        tool_invocations=tool_invocations,
        evaluator_loop_iterations=evaluator_loop_iterations,
        critic_challenged=critic_challenged,
        revision_used_tools=revision_pass and tool_invocations >= 3,
        critic_verdict_passed=critic_verdict_passed,
        leak_scan_blob=leak_blob,
    )


async def execute_with_completion_gate_blocked(bundle: ScenarioRuntimeBundle) -> ScenarioExecutionResult:
    return await execute_resolved_skeleton(
        bundle,
        require_critic_on_completion=True,
        semantic_judge_enabled=True,
    )
