# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Autonomous Work durable state repository ports and adapters (AW-2A)."""

from __future__ import annotations

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
    "AutonomousWorkEntityConflict",
    "AutonomousWorkEntityNotFound",
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
)
