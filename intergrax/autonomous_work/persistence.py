# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-root factories for Autonomous Work repository adapters (AW-2C)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from intergrax.autonomous_work.postgresql_repository import (
    PostgreSQLAutonomousWorkStore,
    PostgreSQLResponsibilityRepository,
    PostgreSQLWorkContinuityStateRepository,
    PostgreSQLWorkerDefinitionRepository,
    PostgreSQLWorkerGoalRepository,
    PostgreSQLWorkerInstanceRepository,
)
from intergrax.autonomous_work.repository import (
    ResponsibilityRepository,
    WorkContinuityStateRepository,
    WorkerDefinitionRepository,
    WorkerGoalRepository,
    WorkerInstanceRepository,
)
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)


@runtime_checkable
class AutonomousWorkStoreOwner(Protocol):
    """Lifecycle owner for durable Autonomous Work repository adapters."""

    def close(self) -> None:
        """Release persistence resources."""


@dataclass(frozen=True, slots=True)
class AutonomousWorkRepositories:
    """Bundle of authoritative Autonomous Work repository ports."""

    worker_definition: WorkerDefinitionRepository
    worker_instance: WorkerInstanceRepository
    responsibility: ResponsibilityRepository
    worker_goal: WorkerGoalRepository
    work_continuity_state: WorkContinuityStateRepository
    store: AutonomousWorkStoreOwner

    def close(self) -> None:
        self.store.close()


def open_postgresql_autonomous_work_repositories(
    *,
    config: PostgreSQLIntegrationConfig | None = None,
    connection_factory: Callable[[], Any] | None = None,
    schema_name: str | None = None,
) -> AutonomousWorkRepositories:
    """Open production-grade Autonomous Work repositories backed by PostgreSQL."""
    resolved = config or PostgreSQLIntegrationConfig.from_env()
    try:
        store = PostgreSQLAutonomousWorkStore(
            resolved,
            connection_factory=connection_factory,
            schema_name=schema_name,
        )
    except IntegrationConfigurationError:
        raise
    except Exception as exc:
        raise IntegrationConfigurationError(
            "PostgreSQL Autonomous Work repositories could not be opened"
        ) from exc
    return AutonomousWorkRepositories(
        worker_definition=PostgreSQLWorkerDefinitionRepository(store),
        worker_instance=PostgreSQLWorkerInstanceRepository(store),
        responsibility=PostgreSQLResponsibilityRepository(store),
        worker_goal=PostgreSQLWorkerGoalRepository(store),
        work_continuity_state=PostgreSQLWorkContinuityStateRepository(store),
        store=store,
    )
