# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""NPSC caller adapter: semantic coordination intent → governance request."""

from __future__ import annotations

from typing import TypeVar

from intergrax.agent_distribution.coordination_binding_materialization import (
    CoordinationCollaborativeApplicabilityClassification,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationExecutionMode,
    CoordinationIntent,
)
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeed,
    AgentDistributionCapabilityNeedKind,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopeId
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationCapabilityKind,
    MultiAgentCoordinationCollaborativeApplicability,
    MultiAgentCoordinationCollaborativeContext,
    MultiAgentCoordinationExecutionMode,
    MultiAgentCoordinationGovernanceContribution,
    MultiAgentCoordinationGovernanceRequest,
    multi_agent_coordination_acting_principal_id,
)

RequestT = TypeVar("RequestT")


def _capability_snapshot(
    capability_need: AgentDistributionCapabilityNeed,
) -> tuple[MultiAgentCoordinationCapabilityKind, tuple[str, ...]]:
    if capability_need.kind is AgentDistributionCapabilityNeedKind.UNRESOLVED_TASK:
        if capability_need.unresolved_task is None:
            raise ValueError("unresolved_task capability need missing payload")
        return MultiAgentCoordinationCapabilityKind.UNRESOLVED_TASK, ()
    if capability_need.resolved_requirement is None:
        raise ValueError("resolved_requirement capability need missing payload")
    return (
        MultiAgentCoordinationCapabilityKind.RESOLVED_REQUIREMENT,
        tuple(
            sorted(
                str(capability_id)
                for capability_id in capability_need.resolved_requirement.required_capability_ids
            ),
        ),
    )


def _execution_mode(mode: CoordinationExecutionMode) -> MultiAgentCoordinationExecutionMode:
    if mode is CoordinationExecutionMode.SINGLE:
        return MultiAgentCoordinationExecutionMode.SINGLE
    if mode is CoordinationExecutionMode.FAN_OUT:
        return MultiAgentCoordinationExecutionMode.FAN_OUT
    raise ValueError(f"unsupported coordination execution mode: {mode}")


def build_multi_agent_coordination_governance_request(
    intent: CoordinationIntent[RequestT],
    *,
    task_scope_id: TaskScopeId,
    application_id: str,
    application_environment_id: str,
    principal: RequestIdentity,
    collaborative_applicability: CoordinationCollaborativeApplicabilityClassification,
) -> MultiAgentCoordinationGovernanceRequest:
    """Project validated semantic intent into governance-owned admission request."""
    contributions: list[MultiAgentCoordinationGovernanceContribution] = []
    for contribution in intent.contributions:
        capability_kind, required_capability_ids = _capability_snapshot(
            contribution.capability_need,
        )
        contributions.append(
            MultiAgentCoordinationGovernanceContribution(
                contribution_id=str(contribution.contribution_id),
                capability_kind=capability_kind,
                required_capability_ids=required_capability_ids,
            ),
        )
    collaborative_context: MultiAgentCoordinationCollaborativeContext | None = None
    if (
        collaborative_applicability.applicability
        is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
    ):
        collaborative_context = MultiAgentCoordinationCollaborativeContext(
            workspace_id=collaborative_applicability.workspace_id or "",
            acting_principal_id=multi_agent_coordination_acting_principal_id(principal),
            delegator_principal_id=collaborative_applicability.delegator_principal_id,
            delegation_id=collaborative_applicability.delegation_id,
            resource_scope=collaborative_applicability.resource_scope,
        )
    return MultiAgentCoordinationGovernanceRequest(
        intent_id=str(intent.intent_id),
        execution_mode=_execution_mode(intent.mode),
        contributions=tuple(contributions),
        requested_max_concurrency=intent.requested_max_concurrency,
        task_scope_id=str(task_scope_id),
        application_id=application_id,
        application_environment_id=application_environment_id,
        principal=principal,
        collaborative_applicability=collaborative_applicability.applicability,
        collaborative_context=collaborative_context,
    )


__all__ = ["build_multi_agent_coordination_governance_request"]
