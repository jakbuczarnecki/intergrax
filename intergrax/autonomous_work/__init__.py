# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Autonomous Work durable state repository ports, adapters, and lifecycle (AW-2A/B)."""

from __future__ import annotations

from intergrax.autonomous_work.lifecycle import (
    AutonomousWorkClock,
    AutonomousWorkInvalidLifecycleTransition,
    AutonomousWorkLifecycleClockError,
    AutonomousWorkLifecycleStateConflict,
    WorkerLifecycleService,
    WorkerLifecycleTransitionPolicy,
    WorkerLifecycleTransitionRequest,
    WorkerLifecycleTransitionResult,
)
from intergrax.autonomous_work.in_memory_repository import (
    InMemoryGoalEvaluationCadenceStateRepository,
    InMemoryResponsibilityRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerDefinitionRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkerPrincipalBindingRepository,
    InMemoryWorkerWakeUpReceiptRepository,
)
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.persistence_provider import resolve_autonomous_work_repositories
from intergrax.autonomous_work.goal_evaluation_ports import (
    DeterministicThresholdGoalEvaluator,
    GoalEvaluationCadenceResolutionError,
    GoalEvaluationCadenceResolver,
    GoalEvaluationCadenceStateStore,
    GoalProgressProjectionResolver,
    InMemoryGoalEvaluationCadenceStateStore,
    MappingGoalEvaluationCadenceResolver,
    MappingGoalProgressProjectionResolver,
    WorkerGoalEvaluator,
)
from intergrax.autonomous_work.goal_evaluation_service import (
    WorkerGoalEvaluationRejected,
    WorkerGoalEvaluationService,
)
from intergrax.autonomous_work.collaborative_work_intake import (
    CollaborativeWorkIntakePort,
    CollaborativeWorkIntakeUnavailable,
    RecordingCollaborativeWorkIntake,
    UnavailableCollaborativeWorkIntake,
)
from intergrax.autonomous_work.execution_authority_admission import (
    CollaborativeWorkAuthorityResolverPort,
    WorkerExecutionAdmissionService,
    WorkerExecutionAuthorityDenied,
)
from intergrax.autonomous_work.worker_collaborative_work_bridge import (
    WorkerCollaborativeWorkBridge,
    WorkerCollaborativeWorkBridgeRejected,
)
from intergrax.autonomous_work.worker_execution_dispatch import (
    WorkerExecutionDispatchService,
)
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingRequired,
    WorkerPrincipalBindingResolver,
)
from intergrax.contracts.autonomous_work.execution_authority import (
    WorkerExecutionAuthorityContext,
    WorkerExecutionAuthorityRequest,
)
from intergrax.contracts.autonomous_work.principal_binding import ResolvedWorkerPrincipal
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkEntityNotFound,
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
    GoalEvaluationCadenceStateRepository,
    ResponsibilityRepository,
    WorkContinuityStateRepository,
    WorkerDefinitionRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
    WorkerPrincipalBindingRepository,
    WorkerWakeUpReceiptRepository,
)
from intergrax.autonomous_work.wake_up_service import (
    WorkerWakeUpEligibilityPolicy,
    WorkerWakeUpPersistenceUnavailable,
    WorkerWakeUpService,
)

__all__ = (
    "AutonomousWorkClock",
    "AutonomousWorkEntityConflict",
    "AutonomousWorkEntityNotFound",
    "AutonomousWorkInvalidLifecycleTransition",
    "AutonomousWorkLifecycleClockError",
    "AutonomousWorkLifecycleStateConflict",
    "AutonomousWorkRepositoryCapabilities",
    "AutonomousWorkRevisionConflict",
    "AutonomousWorkRepositories",
    "DeterministicThresholdGoalEvaluator",
    "GoalEvaluationCadenceResolutionError",
    "GoalEvaluationCadenceResolver",
    "GoalEvaluationCadenceStateRepository",
    "GoalEvaluationCadenceStateStore",
    "GoalProgressProjectionResolver",
    "InMemoryGoalEvaluationCadenceStateRepository",
    "InMemoryGoalEvaluationCadenceStateStore",
    "MappingGoalEvaluationCadenceResolver",
    "MappingGoalProgressProjectionResolver",
    "WorkerGoalEvaluator",
    "WorkerGoalEvaluationRejected",
    "WorkerGoalEvaluationService",
    "CollaborativeWorkAuthorityResolverPort",
    "CollaborativeWorkIntakePort",
    "CollaborativeWorkIntakeUnavailable",
    "RecordingCollaborativeWorkIntake",
    "UnavailableCollaborativeWorkIntake",
    "WorkerCollaborativeWorkBridge",
    "WorkerCollaborativeWorkBridgeRejected",
    "WorkerExecutionDispatchService",
    "resolve_autonomous_work_repositories",
    "InMemoryResponsibilityRepository",
    "InMemoryWorkContinuityStateRepository",
    "InMemoryWorkerDefinitionRepository",
    "InMemoryWorkerGoalRepository",
    "InMemoryWorkerInstanceRepository",
    "InMemoryWorkerPrincipalBindingRepository",
    "InMemoryWorkerWakeUpReceiptRepository",
    "ResponsibilityRepository",
    "WorkContinuityStateRepository",
    "WorkerDefinitionRepository",
    "WorkerGoalRepository",
    "WorkerInstanceRepository",
    "WorkerPrincipalBindingRepository",
    "WorkerWakeUpEligibilityPolicy",
    "WorkerWakeUpPersistenceUnavailable",
    "WorkerWakeUpReceiptRepository",
    "WorkerWakeUpService",
    "ResolvedWorkerPrincipal",
    "WorkerExecutionAdmissionService",
    "WorkerExecutionAuthorityContext",
    "WorkerExecutionAuthorityDenied",
    "WorkerExecutionAuthorityRequest",
    "WorkerPrincipalBindingRequired",
    "WorkerPrincipalBindingResolver",
    "WorkerLifecycleService",
    "WorkerLifecycleTransitionPolicy",
    "WorkerLifecycleTransitionRequest",
    "WorkerLifecycleTransitionResult",
)
