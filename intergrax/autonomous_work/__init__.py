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
    "InMemoryResponsibilityRepository",
    "InMemoryWorkContinuityStateRepository",
    "InMemoryWorkerDefinitionRepository",
    "InMemoryWorkerGoalRepository",
    "InMemoryWorkerInstanceRepository",
    "ResponsibilityRepository",
    "WorkContinuityStateRepository",
    "WorkerDefinitionRepository",
    "WorkerGoalRepository",
    "WorkerInstanceRepository",
    "WorkerLifecycleService",
    "WorkerLifecycleTransitionPolicy",
    "WorkerLifecycleTransitionRequest",
    "WorkerLifecycleTransitionResult",
)
