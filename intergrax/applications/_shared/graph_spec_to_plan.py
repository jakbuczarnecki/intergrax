# © Artur Czarnecki. All rights reserved.

"""Convert Tier-3 ``ApplicationGraphSpec`` into a Nexus plan (Phase ORCH-2, FLOW-2/14)."""

from __future__ import annotations

from collections import defaultdict
from uuid import uuid4

from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdgeKind,
)
from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.task.task import Task


def _step_id_for_agent(agent_id: str) -> str:
    return f"node_{agent_id}"


def application_graph_spec_to_nexus_plan(
    spec: ApplicationGraphSpec,
    task: Task,
    *,
    classification: str,
) -> NexusPlan:
    """
    Map declarative application topology to a ``NexusPlan``.

    ``DEPENDS_ON`` edges become ``PlanStep.depends_on``.
    ``DELEGATES_TO`` edges expand per ADR-FLOW-001: child step depends on parent;
    ``DelegationSpec`` (via ``SubtaskContract``) is attached on the **child** step.
    """
    if not spec.nodes:
        raise ValueError("ApplicationGraphSpec must contain at least one node")

    step_ids = {_step_id_for_agent(node.agent_id): node.agent_id for node in spec.nodes}
    depends_on: dict[str, list[str]] = defaultdict(list)
    child_delegations: dict[str, SubtaskContract] = {}

    for edge in spec.edges:
        source_step = _step_id_for_agent(edge.source_agent_id)
        target_step = _step_id_for_agent(edge.target_agent_id)
        if edge.kind is GraphEdgeKind.DEPENDS_ON:
            if source_step not in depends_on[target_step]:
                depends_on[target_step].append(source_step)
        elif edge.kind is GraphEdgeKind.DELEGATES_TO:
            if source_step not in depends_on[target_step]:
                depends_on[target_step].append(source_step)
            child_delegations[target_step] = SubtaskContract(
                child_agent_id=edge.target_agent_id,
                objective=f"delegated subtask from {edge.source_agent_id}",
                inherit_tool_policy=False,
            )

    steps: list[PlanStep] = []
    for node in spec.nodes:
        step_id = _step_id_for_agent(node.agent_id)
        contract = child_delegations.get(step_id)
        delegation = None
        if contract is not None:
            parent_candidates = depends_on.get(step_id, [])
            parent_step_id = parent_candidates[0] if parent_candidates else None
            delegation = contract.to_delegation_spec(parent_node_id=parent_step_id)
        steps.append(
            PlanStep(
                step_id=step_id,
                agent_id=node.agent_id,
                capability=task.context.capability,
                description=f"graph node {node.agent_id}",
                depends_on=list(depends_on.get(step_id, [])),
                delegation=delegation,
            )
        )

    criteria = ["non_empty_summary"]
    if task.context.capability:
        criteria.append(f"capability:{task.context.capability}")

    return NexusPlan(
        plan_id=f"graph_plan_{uuid4().hex}",
        task_id=task.task_id,
        classification=classification or TaskClassification.MULTI_AGENT.value,
        steps=steps,
        validation_criteria=criteria,
        graph_retry_on_error=spec.retry_on_error,
    )


def should_seed_plan_from_graph_spec(task: Task) -> bool:
    """True when the task has no pre-built orchestration plan id."""
    plan_id = task.runtime.orchestration.plan_id
    if plan_id is None:
        return True
    return not plan_id.strip()
