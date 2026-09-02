# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory reference repositories for Autonomous Work durable state (AW-2A)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Generic, TypeVar

from intergrax.autonomous_work.repository import (
    AutonomousWorkEntityConflict,
    AutonomousWorkEntityNotFound,
    AutonomousWorkRepositoryCapabilities,
    AutonomousWorkRevisionConflict,
)
from intergrax.contracts.autonomous_work.continuity import WorkContinuityState
from intergrax.contracts.autonomous_work.goal import WorkerGoal
from intergrax.contracts.autonomous_work.ids import (
    ResponsibilityId,
    WorkerDefinitionId,
    WorkerGoalId,
    WorkerInstanceId,
)
from intergrax.contracts.autonomous_work.responsibility import Responsibility
from intergrax.contracts.autonomous_work.revision import (
    DefinitionRevision,
    Revision,
    initial_revision,
)
from intergrax.contracts.autonomous_work.worker import WorkerDefinition, WorkerInstance

DefinitionVersionKey = tuple[WorkerDefinitionId, DefinitionRevision]

_KeyT = TypeVar("_KeyT")
_EntityT = TypeVar("_EntityT")


class _ImmutableVersionStore(Generic[_KeyT, _EntityT]):
    """Thread-safe idempotent create for immutable versioned records."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[_KeyT, _EntityT] = {}

    def get(self, key: _KeyT) -> _EntityT | None:
        with self._lock:
            stored = self._records.get(key)
            if stored is None:
                return None
            return stored

    def create_idempotent(
        self,
        key: _KeyT,
        entity: _EntityT,
        *,
        entity_kind: str,
        entity_id: str,
    ) -> _EntityT:
        with self._lock:
            current = self._records.get(key)
            if current is None:
                self._records[key] = entity
                return entity
            if current == entity:
                return current
            raise AutonomousWorkEntityConflict(
                f"{entity_kind} already exists with different content for {entity_id}"
            )


class _RevisionedEntityStore(Generic[_KeyT, _EntityT]):
    """Thread-safe idempotent create and CAS replace for revisioned entities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[_KeyT, _EntityT] = {}

    def get(self, key: _KeyT) -> _EntityT | None:
        with self._lock:
            stored = self._records.get(key)
            if stored is None:
                return None
            return stored

    def create_idempotent(
        self,
        key: _KeyT,
        entity: _EntityT,
        *,
        entity_kind: str,
        entity_id: str,
        initial_revision_value: Revision,
        read_revision: Callable[[_EntityT], Revision],
    ) -> _EntityT:
        with self._lock:
            current = self._records.get(key)
            if current is None:
                if read_revision(entity) != initial_revision_value:
                    raise ValueError(
                        f"{entity_kind} create requires revision {initial_revision_value.value}"
                    )
                self._records[key] = entity
                return entity
            if current == entity:
                return current
            raise AutonomousWorkEntityConflict(
                f"{entity_kind} already exists with different content for {entity_id}"
            )

    def replace(
        self,
        key: _KeyT,
        entity: _EntityT,
        *,
        expected_revision: Revision,
        entity_kind: str,
        entity_id: str,
        read_revision: Callable[[_EntityT], Revision],
        write_revision: Callable[[_EntityT, Revision], _EntityT],
    ) -> _EntityT:
        with self._lock:
            current = self._records.get(key)
            if current is None:
                raise AutonomousWorkEntityNotFound(
                    f"{entity_kind} not found for {entity_id}"
                )
            current_revision = read_revision(current)
            if current_revision != expected_revision:
                raise AutonomousWorkRevisionConflict(
                    (
                        f"{entity_kind} revision conflict for {entity_id}: "
                        f"expected {expected_revision.value}, actual {current_revision.value}"
                    ),
                    entity_kind=entity_kind,
                    entity_id=entity_id,
                    expected_revision=expected_revision,
                    actual_revision=current_revision,
                )
            next_revision = Revision(expected_revision.value + 1)
            persisted = write_revision(entity, next_revision)
            self._records[key] = persisted
            return persisted


def _definition_key(definition: WorkerDefinition) -> DefinitionVersionKey:
    return (definition.worker_definition_id, definition.revision)


def _worker_instance_revision(instance: WorkerInstance) -> Revision:
    return instance.revision


def _worker_instance_with_revision(
    instance: WorkerInstance,
    revision: Revision,
) -> WorkerInstance:
    return replace(instance, revision=revision)


def _responsibility_revision(responsibility: Responsibility) -> Revision:
    return responsibility.revision


def _responsibility_with_revision(
    responsibility: Responsibility,
    revision: Revision,
) -> Responsibility:
    return replace(responsibility, revision=revision)


def _worker_goal_revision(goal: WorkerGoal) -> Revision:
    return goal.revision


def _worker_goal_with_revision(goal: WorkerGoal, revision: Revision) -> WorkerGoal:
    return replace(goal, revision=revision)


def _continuity_revision(state: WorkContinuityState) -> Revision:
    return state.revision


def _continuity_with_revision(
    state: WorkContinuityState,
    revision: Revision,
) -> WorkContinuityState:
    return replace(state, revision=revision)


class InMemoryWorkerDefinitionRepository:
    """Process-local reference repository for immutable WorkerDefinition versions."""

    def __init__(self) -> None:
        self._store: _ImmutableVersionStore[
            DefinitionVersionKey, WorkerDefinition
        ] = _ImmutableVersionStore()

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.worker_definition.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, definition: WorkerDefinition) -> WorkerDefinition:
        key = _definition_key(definition)
        entity_id = (
            f"{definition.worker_definition_id}@{definition.revision.value}"
        )
        return self._store.create_idempotent(
            key,
            definition,
            entity_kind="WorkerDefinition",
            entity_id=entity_id,
        )

    def get(
        self,
        *,
        worker_definition_id: WorkerDefinitionId,
        definition_revision: DefinitionRevision,
    ) -> WorkerDefinition | None:
        return self._store.get((worker_definition_id, definition_revision))


class InMemoryWorkerInstanceRepository:
    """Process-local reference repository for durable WorkerInstance records."""

    def __init__(self) -> None:
        self._store: _RevisionedEntityStore[WorkerInstanceId, WorkerInstance] = (
            _RevisionedEntityStore()
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.worker_instance.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, instance: WorkerInstance) -> WorkerInstance:
        return self._store.create_idempotent(
            instance.worker_instance_id,
            instance,
            entity_kind="WorkerInstance",
            entity_id=instance.worker_instance_id,
            initial_revision_value=initial_revision(),
            read_revision=_worker_instance_revision,
        )

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkerInstance | None:
        return self._store.get(worker_instance_id)

    def replace(
        self,
        instance: WorkerInstance,
        *,
        expected_revision: Revision,
    ) -> WorkerInstance:
        return self._store.replace(
            instance.worker_instance_id,
            instance,
            expected_revision=expected_revision,
            entity_kind="WorkerInstance",
            entity_id=instance.worker_instance_id,
            read_revision=_worker_instance_revision,
            write_revision=_worker_instance_with_revision,
        )


class InMemoryResponsibilityRepository:
    """Process-local reference repository for Responsibility records."""

    def __init__(self) -> None:
        self._store: _RevisionedEntityStore[ResponsibilityId, Responsibility] = (
            _RevisionedEntityStore()
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.responsibility.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, responsibility: Responsibility) -> Responsibility:
        return self._store.create_idempotent(
            responsibility.responsibility_id,
            responsibility,
            entity_kind="Responsibility",
            entity_id=responsibility.responsibility_id,
            initial_revision_value=initial_revision(),
            read_revision=_responsibility_revision,
        )

    def get(self, *, responsibility_id: ResponsibilityId) -> Responsibility | None:
        return self._store.get(responsibility_id)

    def replace(
        self,
        responsibility: Responsibility,
        *,
        expected_revision: Revision,
    ) -> Responsibility:
        return self._store.replace(
            responsibility.responsibility_id,
            responsibility,
            expected_revision=expected_revision,
            entity_kind="Responsibility",
            entity_id=responsibility.responsibility_id,
            read_revision=_responsibility_revision,
            write_revision=_responsibility_with_revision,
        )


class InMemoryWorkerGoalRepository:
    """Process-local reference repository for WorkerGoal records."""

    def __init__(self) -> None:
        self._store: _RevisionedEntityStore[WorkerGoalId, WorkerGoal] = (
            _RevisionedEntityStore()
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.worker_goal.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, goal: WorkerGoal) -> WorkerGoal:
        return self._store.create_idempotent(
            goal.goal_id,
            goal,
            entity_kind="WorkerGoal",
            entity_id=goal.goal_id,
            initial_revision_value=initial_revision(),
            read_revision=_worker_goal_revision,
        )

    def get(self, *, goal_id: WorkerGoalId) -> WorkerGoal | None:
        return self._store.get(goal_id)

    def replace(
        self,
        goal: WorkerGoal,
        *,
        expected_revision: Revision,
    ) -> WorkerGoal:
        return self._store.replace(
            goal.goal_id,
            goal,
            expected_revision=expected_revision,
            entity_kind="WorkerGoal",
            entity_id=goal.goal_id,
            read_revision=_worker_goal_revision,
            write_revision=_worker_goal_with_revision,
        )


class InMemoryWorkContinuityStateRepository:
    """Process-local reference repository for WorkContinuityState checkpoints."""

    def __init__(self) -> None:
        self._store: _RevisionedEntityStore[WorkerInstanceId, WorkContinuityState] = (
            _RevisionedEntityStore()
        )

    @property
    def capabilities(self) -> AutonomousWorkRepositoryCapabilities:
        return AutonomousWorkRepositoryCapabilities(
            backend_id="autonomous_work.work_continuity_state.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, state: WorkContinuityState) -> WorkContinuityState:
        return self._store.create_idempotent(
            state.worker_instance_ref,
            state,
            entity_kind="WorkContinuityState",
            entity_id=state.worker_instance_ref,
            initial_revision_value=initial_revision(),
            read_revision=_continuity_revision,
        )

    def get(self, *, worker_instance_id: WorkerInstanceId) -> WorkContinuityState | None:
        return self._store.get(worker_instance_id)

    def replace(
        self,
        state: WorkContinuityState,
        *,
        expected_revision: Revision,
    ) -> WorkContinuityState:
        return self._store.replace(
            state.worker_instance_ref,
            state,
            expected_revision=expected_revision,
            entity_kind="WorkContinuityState",
            entity_id=state.worker_instance_ref,
            read_revision=_continuity_revision,
            write_revision=_continuity_with_revision,
        )
