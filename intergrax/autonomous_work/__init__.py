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
    InMemoryResponsibilityRepository,
    InMemoryWorkContinuityStateRepository,
    InMemoryWorkerDefinitionRepository,
    InMemoryWorkerGoalRepository,
    InMemoryWorkerInstanceRepository,
    InMemoryWorkerPrincipalBindingRepository,
)
from intergrax.autonomous_work.persistence import AutonomousWorkRepositories
from intergrax.autonomous_work.persistence_provider import resolve_autonomous_work_repositories
from intergrax.autonomous_work.principal_binding_resolver import (
    WorkerPrincipalBindingRequired,
    WorkerPrincipalBindingResolver,
)
from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkEntityNotFound,
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
    ResponsibilityRepository,
    WorkContinuityStateRepository,
    WorkerDefinitionRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
    WorkerPrincipalBindingRepository,
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
    "resolve_autonomous_work_repositories",
    "InMemoryResponsibilityRepository",
    "InMemoryWorkContinuityStateRepository",
    "InMemoryWorkerDefinitionRepository",
    "InMemoryWorkerGoalRepository",
    "InMemoryWorkerInstanceRepository",
    "InMemoryWorkerPrincipalBindingRepository",
    "ResponsibilityRepository",
    "WorkContinuityStateRepository",
    "WorkerDefinitionRepository",
    "WorkerGoalRepository",
    "WorkerInstanceRepository",
    "WorkerPrincipalBindingRepository",
    "WorkerPrincipalBindingRequired",
    "WorkerPrincipalBindingResolver",
    "WorkerLifecycleService",
    "WorkerLifecycleTransitionPolicy",
    "WorkerLifecycleTransitionRequest",
    "WorkerLifecycleTransitionResult",
)
