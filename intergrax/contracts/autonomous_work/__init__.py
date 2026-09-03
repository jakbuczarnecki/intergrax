# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Autonomous Work core semantic contracts (AW-1A/AW-1B)."""

from __future__ import annotations

from intergrax.contracts.autonomous_work.continuity import (
    ProgressCheckpoint,
    WorkContinuityState,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityContext,
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.goal import WorkerGoal, WorkerGoalStatus
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
    mint_responsibility_id,
    mint_worker_definition_id,
    mint_worker_goal_id,
    mint_worker_instance_id,
    validate_responsibility_id,
    validate_worker_definition_id,
    validate_worker_goal_id,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.lifecycle import (
    CANONICAL_WORKER_LIFECYCLE_STATES,
    WorkerLifecycleState,
)
from intergrax.contracts.autonomous_work.principal_binding import (
    ResolvedWorkerPrincipal,
    WorkerPrincipalBinding,
)
from intergrax.contracts.autonomous_work.profile_reference import (
    BudgetProfileRef,
    CapabilityProfileRef,
    CodecraftProfileRef,
    CollaborationProfileRef,
    EscalationPolicyRef,
    GovernanceProfileRef,
    MemoryProfileRef,
    ObservabilityProfileRef,
    ProfileVersion,
    RiskProfileRef,
    ScheduleProfileRef,
    initial_profile_version,
)
from intergrax.contracts.autonomous_work.references import (
    ArtifactReference,
    ContextAnchorReference,
    DeadlineOrCadenceRef,
    DefaultGoalPolicyRef,
    EvaluationCadenceRef,
    ExternalDependencyReference,
    HumanPendingReference,
    MetricRef,
    PrincipalBindingPolicyRef,
    PrincipalBindingRef,
    ProblemReference,
    ProgressCheckpointRef,
    ProgressProjectionRef,
    ResponsibilityScopeRef,
    ResponsibilityTemplateRef,
    SlaSloRef,
    SuccessCriteriaRef,
    WorkReference,
    WorkspaceContextRef,
    WorkspaceScopeRef,
)
from intergrax.contracts.autonomous_work.responsibility import (
    Responsibility,
    ResponsibilityStatus,
)
from intergrax.contracts.autonomous_work.revision import (
    DefinitionRevision,
    Revision,
    initial_definition_revision,
    initial_revision,
)
from intergrax.contracts.autonomous_work.worker import WorkerDefinition, WorkerInstance

__all__ = (
    "ArtifactReference",
    "BudgetProfileRef",
    "CANONICAL_WORKER_LIFECYCLE_STATES",
    "CapabilityProfileRef",
    "CodecraftProfileRef",
    "CollaborationProfileRef",
    "ContextAnchorReference",
    "DeadlineOrCadenceRef",
    "DefaultGoalPolicyRef",
    "DefinitionRevision",
    "EscalationPolicyRef",
    "EvaluationCadenceRef",
    "ExternalDependencyReference",
    "GovernanceProfileRef",
    "HumanPendingReference",
    "MemoryProfileRef",
    "MetricRef",
    "ObservabilityProfileRef",
    "PrincipalBindingPolicyRef",
    "PrincipalBindingRef",
    "ProfileVersion",
    "ProblemReference",
    "ProgressCheckpoint",
    "ProgressCheckpointRef",
    "ProgressProjectionRef",
    "Responsibility",
    "ResponsibilityId",
    "ResponsibilityScopeRef",
    "ResponsibilityStatus",
    "ResponsibilityTemplateRef",
    "ResolvedWorkerPrincipal",
    "Revision",
    "RiskProfileRef",
    "ScheduleProfileRef",
    "SlaSloRef",
    "SuccessCriteriaRef",
    "WorkContinuityState",
    "WorkerExecutionAuthorityContext",
    "WorkerExecutionAuthorityRequest",
    "WorkReference",
    "WorkerDefinition",
    "WorkerDefinitionId",
    "WorkerGoal",
    "WorkerGoalId",
    "WorkerGoalStatus",
    "WorkerInstance",
    "WorkerInstanceId",
    "WorkerPrincipalBinding",
    "WorkerLifecycleState",
    "WorkspaceContextRef",
    "WorkspaceScopeRef",
    "initial_definition_revision",
    "initial_profile_version",
    "initial_revision",
    "mint_responsibility_id",
    "mint_worker_definition_id",
    "mint_worker_goal_id",
    "mint_worker_instance_id",
    "validate_responsibility_id",
    "validate_worker_definition_id",
    "validate_worker_goal_id",
    "validate_worker_instance_id",
)
