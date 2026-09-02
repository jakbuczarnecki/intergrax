# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Autonomous Work durable state repository contracts (AW-2A).

Provider-neutral persistence ports for WorkerDefinition, WorkerInstance,
Responsibility, WorkerGoal, and WorkContinuityState records.

Revision ownership
------------------
The repository is authoritative for ``Revision`` advancement on mutable
entities. Callers supply replacement semantic values and ``expected_revision``
on replace; the repository writes an immutable replacement record at
``expected_revision + 1``. Callers must not supply arbitrary revision numbers
on create or replace.

WorkerDefinition records are versioned by ``DefinitionRevision`` and are
immutable once stored. Different definition revisions may coexist for the same
``WorkerDefinitionId``.

Concurrency
-----------
Replacing a mutable record requires ``expected_revision`` equal to the stored
revision. A mismatch raises ``AutonomousWorkRevisionConflict`` and leaves stored
state unchanged.

Create semantics
----------------
* ID absent → create succeeds.
* Same identity and identical content → deterministic idempotent success.
* Same identity and different content → ``AutonomousWorkEntityConflict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.goal import WorkerGoal
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
)
from intergrax.contracts.autonomous_work.responsibility import Responsibility
from intergrax.contracts.autonomous_work.revision import DefinitionRevision, Revision
from intergrax.contracts.autonomous_work.worker import WorkerDefinition, WorkerInstance


class AutonomousWorkEntityNotFound(Exception):
    """Entity was not found for the requested identity."""


class AutonomousWorkEntityConflict(Exception):
    """Create conflict — same identity with different semantic content."""


class AutonomousWorkRevisionConflict(Exception):
    """Optimistic revision conflict for a mutable Autonomous Work entity."""

    def __init__(
        self,
        message: str,
        *,
        entity_kind: str,
        entity_id: str,
        expected_revision: Revision,
        actual_revision: Revision,
    ) -> None:
        super().__init__(message)
        self.entity_kind = entity_kind
        self.entity_id = entity_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision


@dataclass(frozen=True, slots=True)
class AutonomousWorkRepositoryCapabilities:
    """Declared backend capabilities for Autonomous Work repositories."""

    backend_id: str
    durable: bool
    reference_only: bool

    def __post_init__(self) -> None:
        if not self.backend_id.strip():
            raise ValueError("backend_id must be non-empty")


@runtime_checkable
class WorkerDefinitionRepository(Protocol):
    """Authoritative persistence port for immutable WorkerDefinition versions."""

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""
        ...

    def create(self, definition: WorkerDefinition) -> WorkerDefinition:
        """Create a definition version or return an identical stored version."""
        ...

    def get(
        self,
        *,
        worker_definition_id: WorkerDefinitionId,
        definition_revision: DefinitionRevision,
    ) -> WorkerDefinition | None:
        """Return the exact definition version or ``None``."""
        ...


@runtime_checkable
class WorkerInstanceRepository(Protocol):
    """Authoritative persistence port for durable WorkerInstance records."""

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""
        ...

    def create(self, instance: WorkerInstance) -> WorkerInstance:
        """Create a worker instance at the initial revision."""
        ...

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance | None:
        """Return the worker instance or ``None``."""
        ...

    def replace(
        self,
        instance: WorkerInstance,
        *,
        expected_revision: Revision,
    ) -> WorkerInstance:
        """Replace worker instance semantics under optimistic concurrency."""
        ...


@runtime_checkable
class ResponsibilityRepository(Protocol):
    """Authoritative persistence port for Responsibility records."""

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""
        ...

    def create(self, responsibility: Responsibility) -> Responsibility:
        """Create a responsibility at the initial revision."""
        ...

    def get(self, *, responsibility_id: ResponsibilityId) -> Responsibility | None:
        """Return the responsibility or ``None``."""
        ...

    def replace(
        self,
        responsibility: Responsibility,
        *,
        expected_revision: Revision,
    ) -> Responsibility:
        """Replace responsibility semantics under optimistic concurrency."""
        ...


@runtime_checkable
class WorkerGoalRepository(Protocol):
    """Authoritative persistence port for WorkerGoal records."""

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""
        ...

    def create(self, goal: WorkerGoal) -> WorkerGoal:
        """Create a worker goal at the initial revision."""
        ...

    def get(self, *, goal_id: WorkerGoalId) -> WorkerGoal | None:
        """Return the worker goal or ``None``."""
        ...

    def replace(
        self,
        goal: WorkerGoal,
        *,
        expected_revision: Revision,
    ) -> WorkerGoal:
        """Replace worker goal semantics under optimistic concurrency."""
        ...


@runtime_checkable
class WorkContinuityStateRepository(Protocol):
    """Authoritative persistence port for WorkContinuityState checkpoints."""

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""
        ...

    def create(self, state: WorkContinuityState) -> WorkContinuityState:
        """Create continuity state for a worker instance at the initial revision."""
        ...

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkContinuityState | None:
        """Return the latest committed continuity state or ``None``."""
        ...

    def replace(
        self,
        state: WorkContinuityState,
        *,
        expected_revision: Revision,
    ) -> WorkContinuityState:
        """Replace continuity state under optimistic concurrency."""
        ...
