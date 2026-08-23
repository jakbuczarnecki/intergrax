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
from intergrax.contracts.evidence_claims import EvidenceChallenge, EvidenceClaimSet
from intergrax.runtime.critic.contracts import CriticVerdict
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
from intergrax.runtime.critic.trace import CriticTraceEmitter
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.tools.registry import ToolRegistry
from platform_proofs.scenarios.ai_incident_investigation.critic_adapter import (
    apply_challenge_lifecycle,
    first_failed_node_partial_verdict_from_trace,
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
)

INVESTIGATOR_NODE_ID = "investigator-1"
OUTCOME_RESOLVED = "RESOLVED"
OUTCOME_UNRESOLVED = "UNRESOLVED"
EVALUATOR_LOOP_MAX_ITERATIONS = 2


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
    failed_critic_verdict: CriticVerdict | None
    evidence_challenge: EvidenceChallenge | None


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


async def execute_resolved_skeleton(
    bundle: ScenarioRuntimeBundle,
    *,
    require_critic_on_completion: bool = True,
    semantic_judge_enabled: bool = False,
    validation_engine: IncidentInvestigationValidationEngine | None = None,
    evaluator_loop_max_iterations: int = EVALUATOR_LOOP_MAX_ITERATIONS,
) -> ScenarioExecutionResult:
    agent_registry = AgentRegistry()
    agent_registry.register(bundle.investigator)

    resolved_validation_engine = validation_engine or IncidentInvestigationValidationEngine()
    critic_hooks = build_critic_graph_hooks(
        config=CriticHookConfig(
            verify_node_partial=True,
            verify_graph_final=True,
            require_critic_on_completion=require_critic_on_completion,
            semantic_judge_enabled=semantic_judge_enabled,
        ),
        validation_engine=resolved_validation_engine,
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
            max_iterations=evaluator_loop_max_iterations,
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
        validation_engine=resolved_validation_engine,
        critic_graph_hooks=critic_hooks,
    )

    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    critic_trace = CriticTraceEmitter(run_id=run_id)
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        executions, _retries, graph_out, _ = await executor.execute(
            graph,
            task,
            critic_trace_emitter=critic_trace,
        )
    finally:
        reset_active_execution_identity(token)

    if not executions:
        node = graph_out.node_by_id(INVESTIGATOR_NODE_ID)
        if node.execution_result is None:
            raise RuntimeError("no agent executions produced")
        final_execution = node.execution_result
    else:
        final_execution = executions[-1]
    domain_payload = domain_payload_from_execution(final_execution)
    claim_set = dict(domain_payload.get("claim_set", {}))
    evidence_nodes = tuple(domain_payload.get("evidence_nodes", []))
    tool_invocations = int(domain_payload.get("tool_invocations", 0))
    revision_pass = bool(domain_payload.get("revision_pass", False))
    evaluator_loop_iterations = current_evaluator_loop_iteration(worker)

    failed_critic_verdict = first_failed_node_partial_verdict_from_trace(
        critic_trace,
        node_id=INVESTIGATOR_NODE_ID,
    )
    critic_challenged = failed_critic_verdict is not None and not failed_critic_verdict.passed

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

    evidence_challenge: EvidenceChallenge | None = None
    if critic_challenged and failed_critic_verdict is not None:
        claim_set, evidence_challenge = apply_challenge_lifecycle(
            claim_set,
            failed_critic_verdict,
            claim_id=INITIAL_CLAIM_ID,
            evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
            resolved=critic_verdict_passed,
        )

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
        failed_critic_verdict=failed_critic_verdict,
        evidence_challenge=evidence_challenge,
    )


async def execute_with_completion_gate_blocked(bundle: ScenarioRuntimeBundle) -> ScenarioExecutionResult:
    """Real graph path where critic failure cannot recover within platform loop budget."""
    return await execute_resolved_skeleton(
        bundle,
        require_critic_on_completion=True,
        evaluator_loop_max_iterations=1,
    )
