# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative coordination binding materialization (NPSC-5D/R1-H2).

Collaborative applicability is a platform fact derived from governed host task
context — not caller omission of optional workspace fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.agent_distribution.task_scoped_agents import TaskScopeId
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationCollaborativeApplicability,
)

if TYPE_CHECKING:
    from intergrax.agent_distribution.coordination_intent_executor import (
        CoordinationContributionBinding,
        CoordinationIntentBinding,
    )
    from intergrax.runtime.task.task import Task


class CoordinationCollaborativeApplicabilityIndeterminateError(Exception):
    """Authoritative host context cannot classify collaborative applicability."""


@dataclass(frozen=True, slots=True)
class CoordinationCollaborativeDelegationLocators:
    """Collaborative delegation locators supplied by host context — not authority proof."""

    delegator_principal_id: str
    delegation_id: str
    resource_scope: str | None = None

    def __post_init__(self) -> None:
        if not self.delegator_principal_id.strip():
            raise ValueError("delegator_principal_id must be non-empty")
        if not self.delegation_id.strip():
            raise ValueError("delegation_id must be non-empty")
        if self.resource_scope is not None and not self.resource_scope.strip():
            raise ValueError("resource_scope must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class CoordinationCollaborativeApplicabilityClassification:
    """Authoritative collaborative applicability carried on runtime binding."""

    applicability: MultiAgentCoordinationCollaborativeApplicability
    workspace_id: str | None = None
    delegator_principal_id: str | None = None
    delegation_id: str | None = None
    resource_scope: str | None = None

    def __post_init__(self) -> None:
        if (
            self.applicability
            is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
        ):
            if self.workspace_id is None or not self.workspace_id.strip():
                raise ValueError(
                    "workspace_id required when collaborative applicability is REQUIRED",
                )
            return
        if self.workspace_id is not None:
            raise ValueError(
                "workspace_id forbidden when collaborative applicability is NOT_APPLICABLE",
            )
        if self.delegator_principal_id is not None:
            raise ValueError(
                "delegator_principal_id forbidden when collaborative applicability is NOT_APPLICABLE",
            )
        if self.delegation_id is not None:
            raise ValueError(
                "delegation_id forbidden when collaborative applicability is NOT_APPLICABLE",
            )
        if self.resource_scope is not None:
            raise ValueError(
                "resource_scope forbidden when collaborative applicability is NOT_APPLICABLE",
            )

    @classmethod
    def not_applicable(cls) -> CoordinationCollaborativeApplicabilityClassification:
        return cls(
            applicability=MultiAgentCoordinationCollaborativeApplicability.NOT_APPLICABLE,
        )

    @classmethod
    def required(
        cls,
        *,
        workspace_id: str,
        delegator_principal_id: str | None = None,
        delegation_id: str | None = None,
        resource_scope: str | None = None,
    ) -> CoordinationCollaborativeApplicabilityClassification:
        normalized_workspace = workspace_id.strip()
        if not normalized_workspace:
            raise ValueError("workspace_id must be non-empty")
        normalized_delegator = (
            delegator_principal_id.strip()
            if delegator_principal_id is not None
            else None
        )
        normalized_delegation = (
            delegation_id.strip() if delegation_id is not None else None
        )
        normalized_resource = (
            resource_scope.strip() if resource_scope is not None else None
        )
        if normalized_delegator is not None and normalized_delegation is None:
            raise ValueError("delegation_id required when delegator_principal_id is set")
        if normalized_delegation is not None and normalized_delegator is None:
            raise ValueError("delegator_principal_id required when delegation_id is set")
        return cls(
            applicability=MultiAgentCoordinationCollaborativeApplicability.REQUIRED,
            workspace_id=normalized_workspace,
            delegator_principal_id=normalized_delegator,
            delegation_id=normalized_delegation,
            resource_scope=normalized_resource,
        )


def _workspace_id_from_governed_task(governed_task: Task) -> str | None:
    raw_workspace = governed_task.metadata.get("workspace_id")
    if raw_workspace is None:
        return None
    normalized = str(raw_workspace).strip()
    if not normalized:
        raise CoordinationCollaborativeApplicabilityIndeterminateError(
            "indeterminate_workspace_context",
        )
    return normalized


def classify_coordination_collaborative_applicability_from_governed_task(
    governed_task: Task | None,
    *,
    delegation_locators: CoordinationCollaborativeDelegationLocators | None = None,
) -> CoordinationCollaborativeApplicabilityClassification:
    """Classify collaborative applicability from canonical governed host task."""
    if governed_task is None:
        if delegation_locators is not None:
            raise CoordinationCollaborativeApplicabilityIndeterminateError(
                "delegation_locators_without_governed_task",
            )
        raise CoordinationCollaborativeApplicabilityIndeterminateError(
            "governed_task_required_for_authoritative_classification",
        )

    workspace_id = _workspace_id_from_governed_task(governed_task)
    if workspace_id is None:
        if delegation_locators is not None:
            raise CoordinationCollaborativeApplicabilityIndeterminateError(
                "delegation_locators_without_workspace_context",
            )
        return CoordinationCollaborativeApplicabilityClassification.not_applicable()

    return CoordinationCollaborativeApplicabilityClassification.required(
        workspace_id=workspace_id,
        delegator_principal_id=(
            delegation_locators.delegator_principal_id
            if delegation_locators is not None
            else None
        ),
        delegation_id=(
            delegation_locators.delegation_id
            if delegation_locators is not None
            else None
        ),
        resource_scope=(
            delegation_locators.resource_scope
            if delegation_locators is not None
            else None
        ),
    )


def reconcile_coordination_collaborative_applicability(
    binding_classification: CoordinationCollaborativeApplicabilityClassification,
    governed_task: Task | None,
) -> str | None:
    """Return fail-closed reason when binding disagrees with authoritative host context."""
    if governed_task is None:
        return "collaborative_applicability_without_authoritative_host_context"

    try:
        authoritative = classify_coordination_collaborative_applicability_from_governed_task(
            governed_task,
        )
    except CoordinationCollaborativeApplicabilityIndeterminateError:
        return "collaborative_applicability_indeterminate"

    if binding_classification.applicability != authoritative.applicability:
        return "collaborative_applicability_authoritative_mismatch"
    if (
        authoritative.applicability
        is MultiAgentCoordinationCollaborativeApplicability.REQUIRED
        and binding_classification.workspace_id != authoritative.workspace_id
    ):
        return "collaborative_workspace_authoritative_mismatch"
    return None


def materialize_coordination_intent_binding(
    *,
    task_scope_id: TaskScopeId,
    application_id: str,
    application_environment_id: str,
    contribution_bindings: tuple[CoordinationContributionBinding, ...],
    governed_task: Task | None = None,
    delegation_locators: CoordinationCollaborativeDelegationLocators | None = None,
) -> CoordinationIntentBinding:
    """Canonical production binding materialization from authoritative host context."""
    from intergrax.agent_distribution.coordination_intent_executor import (
        CoordinationIntentBinding,
    )

    collaborative_applicability = (
        classify_coordination_collaborative_applicability_from_governed_task(
            governed_task,
            delegation_locators=delegation_locators,
        )
    )
    return CoordinationIntentBinding(
        task_scope_id=task_scope_id,
        application_id=application_id,
        application_environment_id=application_environment_id,
        contribution_bindings=contribution_bindings,
        collaborative_applicability=collaborative_applicability,
    )


__all__ = [
    "CoordinationCollaborativeApplicabilityClassification",
    "CoordinationCollaborativeApplicabilityIndeterminateError",
    "CoordinationCollaborativeDelegationLocators",
    "classify_coordination_collaborative_applicability_from_governed_task",
    "materialize_coordination_intent_binding",
    "reconcile_coordination_collaborative_applicability",
]
