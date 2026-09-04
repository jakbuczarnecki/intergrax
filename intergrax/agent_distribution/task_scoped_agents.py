# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task-scoped agent lease ownership over canonical AC-3 lifecycle (AC-4 Phase 7)."""

from __future__ import annotations

from enum import StrEnum
from threading import RLock
from typing import Final, NewType, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BuildApplicationRevisionRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity
from intergrax.agent_distribution.dynamic_acquisition import (
    AgentPlatformLifecyclePort,
    DynamicAgentAcquisitionOutcome,
    DynamicAgentAcquisitionRequest,
    DynamicAgentAcquisitionResult,
)
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import TaskId, validate_task_id

_NON_EMPTY = Field(min_length=1)

SCHEMA_TASK_SCOPED_AGENT_LEASE_V1: Final = "task_scoped_agent_lease.v1"
SCHEMA_TASK_SCOPED_ACQUISITION_REQUEST_V1: Final = (
    "task_scoped_agent_acquisition_request.v1"
)
SCHEMA_TASK_SCOPED_ACQUISITION_RESULT_V1: Final = (
    "task_scoped_agent_acquisition_result.v1"
)
SCHEMA_TASK_SCOPED_RELEASE_REQUEST_V1: Final = "task_scoped_agent_release_request.v1"
SCHEMA_TASK_SCOPED_RELEASE_RESULT_V1: Final = "task_scoped_agent_release_result.v1"

TaskScopeId = TaskId
TaskScopedAgentLeaseId = NewType("TaskScopedAgentLeaseId", str)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def _validate_task_scope_id(value: object) -> TaskScopeId:
    return validate_task_id(value)


def _validate_lease_id(value: object) -> TaskScopedAgentLeaseId:
    if type(value) is not str:
        raise TypeError("lease_id must be str")
    return TaskScopedAgentLeaseId(_strip_required(value))


class TaskScopedAgentError(AgentDistributionError):
    """Base error for task-scoped agent ownership."""


class TaskScopedAgentContractError(TaskScopedAgentError):
    """Malformed task-scoped agent request."""


class TaskScopedAgentLeaseNotFound(TaskScopedAgentError):
    """Lease authority record does not exist."""


class TaskScopedAgentLeaseConflict(TaskScopedAgentError):
    """Lease idempotency or state transition conflict."""


class TaskScopedAgentOwnershipError(TaskScopedAgentError):
    """Caller does not own the lease scope."""


class TaskScopedAgentReleaseError(TaskScopedAgentError):
    """Release desired-state or activation mutation failed."""


class TaskScopedOwnershipMode(StrEnum):
    """Ownership mode supported in Phase 7."""

    TASK_SCOPED = "task_scoped"


class TaskScopedAgentLeaseState(StrEnum):
    """Explicit lease authority states."""

    ACTIVE = "active"
    RELEASE_REQUESTED = "release_requested"
    RELEASE_FAILED = "release_failed"
    RELEASED = "released"


class BindingTaskOrigin(StrEnum):
    """Whether the binding pre-existed the first task-scoped lease."""

    PRE_EXISTING = "pre_existing"
    TASK_CREATED = "task_created"


class TaskScopedAgentAcquisitionOutcome(StrEnum):
    """Task-scoped acquisition terminal semantics."""

    LEASE_ACQUIRED = "lease_acquired"
    LEASE_REUSED = "lease_reused"


class TaskScopedAgentReleaseOutcome(StrEnum):
    """Task-scoped release terminal semantics."""

    LEASE_RELEASED_RETAINED_BINDING = "lease_released_retained_binding"
    LEASE_RELEASED_RUNTIME_UPDATED = "lease_released_runtime_updated"
    ALREADY_RELEASED = "already_released"
    RELEASE_RUNTIME_FAILED = "release_runtime_failed"


class TaskScopedAgentLease(BaseModel):
    """Immutable task-scoped ownership authority for one canonical binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_SCOPED_AGENT_LEASE_V1
    lease_id: TaskScopedAgentLeaseId
    task_scope_id: TaskScopeId
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    ownership_mode: TaskScopedOwnershipMode = TaskScopedOwnershipMode.TASK_SCOPED
    selected_identity: AgentDiscoveryCandidateIdentity
    installation_id: str = _NON_EMPTY
    application_binding_id: str = _NON_EMPTY
    acquisition_runtime_revision_id: str = _NON_EMPTY
    binding_created_by_task: bool
    lease_state: TaskScopedAgentLeaseState
    release_runtime_revision_id: str | None = None

    @field_validator("lease_id", mode="before")
    @classmethod
    def _validate_lease_id_field(cls, value: object) -> TaskScopedAgentLeaseId:
        return _validate_lease_id(value)

    @field_validator("task_scope_id", mode="before")
    @classmethod
    def _validate_task_scope_field(cls, value: object) -> TaskScopeId:
        return _validate_task_scope_id(value)

    @field_validator(
        "application_id",
        "application_environment_id",
        "installation_id",
        "application_binding_id",
        "acquisition_runtime_revision_id",
        "release_runtime_revision_id",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class TaskScopedAgentAcquisitionRequest(BaseModel):
    """Acquire one task-scoped lease via Phase 6 dynamic acquisition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_SCOPED_ACQUISITION_REQUEST_V1
    lease_id: TaskScopedAgentLeaseId
    task_scope_id: TaskScopeId
    ownership_mode: TaskScopedOwnershipMode = TaskScopedOwnershipMode.TASK_SCOPED
    acquisition_request: DynamicAgentAcquisitionRequest

    @field_validator("lease_id", mode="before")
    @classmethod
    def _validate_lease_id_field(cls, value: object) -> TaskScopedAgentLeaseId:
        return _validate_lease_id(value)

    @field_validator("task_scope_id", mode="before")
    @classmethod
    def _validate_task_scope_field(cls, value: object) -> TaskScopeId:
        return _validate_task_scope_id(value)

    @field_validator("ownership_mode")
    @classmethod
    def _validate_mode(cls, value: TaskScopedOwnershipMode) -> TaskScopedOwnershipMode:
        if value is not TaskScopedOwnershipMode.TASK_SCOPED:
            raise ValueError("only task_scoped ownership is supported in phase 7")
        return value


class TaskScopedAgentAcquisitionResult(BaseModel):
    """Lease authority plus underlying Phase 6 acquisition evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_SCOPED_ACQUISITION_RESULT_V1
    outcome: TaskScopedAgentAcquisitionOutcome
    lease: TaskScopedAgentLease
    acquisition_result: DynamicAgentAcquisitionResult


class TaskScopedAgentReleaseRequest(BaseModel):
    """Release one task-scoped lease and reconcile binding ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_SCOPED_RELEASE_REQUEST_V1
    lease_id: TaskScopedAgentLeaseId
    task_scope_id: TaskScopeId
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    disable: SetAgentEnablementRequest
    build: BuildApplicationRevisionRequest
    activate: ActivateRuntimeRevisionRequest

    @field_validator("lease_id", mode="before")
    @classmethod
    def _validate_lease_id_field(cls, value: object) -> TaskScopedAgentLeaseId:
        return _validate_lease_id(value)

    @field_validator("task_scope_id", mode="before")
    @classmethod
    def _validate_task_scope_field(cls, value: object) -> TaskScopeId:
        return _validate_task_scope_id(value)

    @field_validator("application_id", "application_environment_id")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)


class TaskScopedAgentReleaseResult(BaseModel):
    """Release outcome with optional runtime revision evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_SCOPED_RELEASE_RESULT_V1
    outcome: TaskScopedAgentReleaseOutcome
    lease: TaskScopedAgentLease
    traffic_serving_revision_id: str | None = None

    @field_validator("traffic_serving_revision_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)


class TaskScopedAgentLeaseStore(Protocol):
    """Replaceable authority for task-scoped ownership leases only."""

    def get(self, lease_id: TaskScopedAgentLeaseId) -> TaskScopedAgentLease | None: ...

    def put_new(self, lease: TaskScopedAgentLease) -> None: ...

    def compare_and_set(
        self,
        lease_id: TaskScopedAgentLeaseId,
        *,
        expected_state: TaskScopedAgentLeaseState,
        new_lease: TaskScopedAgentLease,
    ) -> bool: ...

    def list_active_by_binding(
        self,
        application_binding_id: str,
    ) -> tuple[TaskScopedAgentLease, ...]: ...

    def list_active_by_task_scope(
        self,
        task_scope_id: TaskScopeId,
    ) -> tuple[TaskScopedAgentLease, ...]: ...

    def get_binding_task_origin(
        self,
        application_binding_id: str,
    ) -> BindingTaskOrigin | None: ...


class DynamicAgentAcquisitionPort(Protocol):
    """Phase 6 acquisition facade — composition boundary only."""

    def acquire(
        self,
        request: DynamicAgentAcquisitionRequest,
        *,
        principal: RequestIdentity,
    ) -> DynamicAgentAcquisitionResult: ...


class TaskScopedAgentLifecyclePort(AgentPlatformLifecyclePort, Protocol):
    """AC-3 lifecycle facade extended with binding disable for release."""

    def disable_binding(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        application_binding_id: str,
        request: SetAgentEnablementRequest,
        principal: RequestIdentity,
    ) -> object: ...


def _lease_matches_idempotent_reacquire(
    existing: TaskScopedAgentLease,
    request: TaskScopedAgentAcquisitionRequest,
) -> bool:
    acquisition = request.acquisition_request
    return (
        existing.task_scope_id == request.task_scope_id
        and existing.application_id == acquisition.application_id
        and existing.application_environment_id
        == acquisition.application_environment_id
        and existing.selected_identity == acquisition.selected_identity
        and existing.installation_id == acquisition.install.installation_id
        and existing.application_binding_id == acquisition.bind.application_binding_id
        and existing.ownership_mode == request.ownership_mode
    )


def _assert_lease_scope(
    lease: TaskScopedAgentLease,
    *,
    task_scope_id: TaskScopeId,
    application_id: str,
    application_environment_id: str,
) -> None:
    if lease.task_scope_id != task_scope_id:
        raise TaskScopedAgentOwnershipError(
            "lease task_scope_id does not match release request",
        )
    if lease.application_id != application_id:
        raise TaskScopedAgentOwnershipError(
            "lease application_id does not match release request",
        )
    if lease.application_environment_id != application_environment_id:
        raise TaskScopedAgentOwnershipError(
            "lease application_environment_id does not match release request",
        )


def binding_requires_runtime_release(
    *,
    lease_store: TaskScopedAgentLeaseStore,
    application_binding_id: str,
    excluding_lease_id: TaskScopedAgentLeaseId,
) -> bool:
    """Return True when the binding may be disabled after releasing one lease."""
    remaining = tuple(
        item
        for item in lease_store.list_active_by_binding(application_binding_id)
        if item.lease_id != excluding_lease_id
    )
    if remaining:
        return False
    origin = lease_store.get_binding_task_origin(application_binding_id)
    if origin is BindingTaskOrigin.PRE_EXISTING:
        return False
    return True


class InMemoryTaskScopedAgentLeaseStore:
    """Process-local lease authority — restart loses leases (Reference Production V1)."""

    def __init__(self) -> None:
        self._leases: dict[TaskScopedAgentLeaseId, TaskScopedAgentLease] = {}
        self._binding_origins: dict[str, BindingTaskOrigin] = {}
        self._lock = RLock()

    def get(self, lease_id: TaskScopedAgentLeaseId) -> TaskScopedAgentLease | None:
        with self._lock:
            return self._leases.get(lease_id)

    def put_new(self, lease: TaskScopedAgentLease) -> None:
        with self._lock:
            existing = self._leases.get(lease.lease_id)
            if existing is not None:
                if existing == lease:
                    return
                raise TaskScopedAgentLeaseConflict(
                    "lease_id already exists with conflicting authority",
                )
            self._leases[lease.lease_id] = lease
            if lease.application_binding_id not in self._binding_origins:
                self._binding_origins[lease.application_binding_id] = (
                    BindingTaskOrigin.TASK_CREATED
                    if lease.binding_created_by_task
                    else BindingTaskOrigin.PRE_EXISTING
                )

    def compare_and_set(
        self,
        lease_id: TaskScopedAgentLeaseId,
        *,
        expected_state: TaskScopedAgentLeaseState,
        new_lease: TaskScopedAgentLease,
    ) -> bool:
        with self._lock:
            current = self._leases.get(lease_id)
            if current is None or current.lease_state is not expected_state:
                return False
            if current.lease_id != new_lease.lease_id:
                raise TaskScopedAgentLeaseConflict("lease_id mismatch on transition")
            self._leases[lease_id] = new_lease
            return True

    def list_active_by_binding(
        self,
        application_binding_id: str,
    ) -> tuple[TaskScopedAgentLease, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        lease
                        for lease in self._leases.values()
                        if lease.application_binding_id == application_binding_id
                        and lease.lease_state is TaskScopedAgentLeaseState.ACTIVE
                    ),
                    key=lambda item: str(item.lease_id),
                ),
            )

    def list_active_by_task_scope(
        self,
        task_scope_id: TaskScopeId,
    ) -> tuple[TaskScopedAgentLease, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        lease
                        for lease in self._leases.values()
                        if lease.task_scope_id == task_scope_id
                        and lease.lease_state is TaskScopedAgentLeaseState.ACTIVE
                    ),
                    key=lambda item: str(item.lease_id),
                ),
            )

    def get_binding_task_origin(
        self,
        application_binding_id: str,
    ) -> BindingTaskOrigin | None:
        with self._lock:
            return self._binding_origins.get(application_binding_id)


class TaskScopedAgentAcquisitionService:
    """Acquire task-scoped leases via Phase 6 without duplicating lifecycle."""

    def __init__(
        self,
        *,
        acquisition: DynamicAgentAcquisitionPort,
        lease_store: TaskScopedAgentLeaseStore,
    ) -> None:
        self._acquisition = acquisition
        self._lease_store = lease_store

    def acquire(
        self,
        request: TaskScopedAgentAcquisitionRequest,
        *,
        principal: RequestIdentity,
    ) -> TaskScopedAgentAcquisitionResult:
        acquisition_request = request.acquisition_request
        existing = self._lease_store.get(request.lease_id)
        if existing is not None:
            return self._reuse_existing_lease(
                existing,
                request,
                principal=principal,
            )

        acquisition_result = self._acquisition.acquire(
            acquisition_request,
            principal=principal,
        )
        if acquisition_result.outcome not in {
            DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE,
            DynamicAgentAcquisitionOutcome.ALREADY_ACTIVE,
        }:
            raise TaskScopedAgentError(
                "canonical acquisition did not reach active serving state",
            )

        lease = TaskScopedAgentLease(
            lease_id=request.lease_id,
            task_scope_id=request.task_scope_id,
            application_id=acquisition_request.application_id,
            application_environment_id=acquisition_request.application_environment_id,
            ownership_mode=request.ownership_mode,
            selected_identity=acquisition_result.selected_identity,
            installation_id=acquisition_result.installation_id,
            application_binding_id=acquisition_result.application_binding_id,
            acquisition_runtime_revision_id=acquisition_result.runtime_revision_id,
            binding_created_by_task=not acquisition_result.binding_reused,
            lease_state=TaskScopedAgentLeaseState.ACTIVE,
        )
        self._lease_store.put_new(lease)
        return TaskScopedAgentAcquisitionResult(
            outcome=TaskScopedAgentAcquisitionOutcome.LEASE_ACQUIRED,
            lease=lease,
            acquisition_result=acquisition_result,
        )

    def _reuse_existing_lease(
        self,
        existing: TaskScopedAgentLease,
        request: TaskScopedAgentAcquisitionRequest,
        *,
        principal: RequestIdentity,
    ) -> TaskScopedAgentAcquisitionResult:
        if not _lease_matches_idempotent_reacquire(existing, request):
            raise TaskScopedAgentLeaseConflict(
                "lease_id reused with conflicting task scope or binding intent",
            )
        if existing.lease_state is not TaskScopedAgentLeaseState.ACTIVE:
            raise TaskScopedAgentLeaseConflict(
                "lease_id exists but is not active for reuse",
            )
        acquisition_result = self._acquisition.acquire(
            request.acquisition_request,
            principal=principal,
        )
        return TaskScopedAgentAcquisitionResult(
            outcome=TaskScopedAgentAcquisitionOutcome.LEASE_REUSED,
            lease=existing,
            acquisition_result=acquisition_result,
        )


class TaskScopedAgentReleaseService:
    """Release task-scoped leases and reconcile canonical binding ownership."""

    def __init__(
        self,
        *,
        lifecycle: TaskScopedAgentLifecyclePort,
        lease_store: TaskScopedAgentLeaseStore,
    ) -> None:
        self._lifecycle = lifecycle
        self._lease_store = lease_store

    def release(
        self,
        request: TaskScopedAgentReleaseRequest,
        *,
        principal: RequestIdentity,
    ) -> TaskScopedAgentReleaseResult:
        lease = self._lease_store.get(request.lease_id)
        if lease is None:
            raise TaskScopedAgentLeaseNotFound(
                f"lease {request.lease_id} not found",
            )
        _assert_lease_scope(
            lease,
            task_scope_id=request.task_scope_id,
            application_id=request.application_id,
            application_environment_id=request.application_environment_id,
        )

        if lease.lease_state is TaskScopedAgentLeaseState.RELEASED:
            return TaskScopedAgentReleaseResult(
                outcome=TaskScopedAgentReleaseOutcome.ALREADY_RELEASED,
                lease=lease,
                traffic_serving_revision_id=None,
            )

        retain_binding = not binding_requires_runtime_release(
            lease_store=self._lease_store,
            application_binding_id=lease.application_binding_id,
            excluding_lease_id=lease.lease_id,
        )

        if retain_binding:
            released = lease.model_copy(
                update={"lease_state": TaskScopedAgentLeaseState.RELEASED},
            )
            if not self._lease_store.compare_and_set(
                lease.lease_id,
                expected_state=lease.lease_state,
                new_lease=released,
            ):
                current = self._lease_store.get(request.lease_id)
                if current is None:
                    raise TaskScopedAgentLeaseNotFound(
                        f"lease {request.lease_id} not found",
                    )
                if current.lease_state is TaskScopedAgentLeaseState.RELEASED:
                    return TaskScopedAgentReleaseResult(
                        outcome=TaskScopedAgentReleaseOutcome.ALREADY_RELEASED,
                        lease=current,
                    )
                raise TaskScopedAgentLeaseConflict(
                    "lease state changed during release",
                )
            return TaskScopedAgentReleaseResult(
                outcome=TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RETAINED_BINDING,
                lease=released,
            )

        release_requested = lease.model_copy(
            update={"lease_state": TaskScopedAgentLeaseState.RELEASE_REQUESTED},
        )
        if lease.lease_state is TaskScopedAgentLeaseState.ACTIVE:
            if not self._lease_store.compare_and_set(
                lease.lease_id,
                expected_state=TaskScopedAgentLeaseState.ACTIVE,
                new_lease=release_requested,
            ):
                return self.release(request, principal=principal)
        elif lease.lease_state not in {
            TaskScopedAgentLeaseState.RELEASE_REQUESTED,
            TaskScopedAgentLeaseState.RELEASE_FAILED,
        }:
            raise TaskScopedAgentLeaseConflict(
                f"lease {lease.lease_id} is not releasable from {lease.lease_state}",
            )

        working = self._lease_store.get(request.lease_id)
        if working is None:
            raise TaskScopedAgentLeaseNotFound(f"lease {request.lease_id} not found")

        try:
            self._lifecycle.disable_binding(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                application_binding_id=working.application_binding_id,
                request=request.disable,
                principal=principal,
            )
            build_result = self._lifecycle.build_application_revision(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                request=request.build,
                principal=principal,
            )
            activate_view = self._lifecycle.activate_revision(
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                request=request.activate.model_copy(
                    update={
                        "runtime_revision_id": request.build.runtime_revision_id,
                        "artifact_locator": build_result.artifact_locator
                        or request.activate.artifact_locator,
                        "expected_artifact_digest": (
                            build_result.materialization_artifact_digest
                            or request.activate.expected_artifact_digest
                        ),
                    },
                ),
                principal=principal,
            )
        except AgentDistributionError as exc:
            failed = working.model_copy(
                update={"lease_state": TaskScopedAgentLeaseState.RELEASE_FAILED},
            )
            self._lease_store.compare_and_set(
                working.lease_id,
                expected_state=working.lease_state,
                new_lease=failed,
            )
            raise TaskScopedAgentReleaseError(
                "binding release desired-state updated but runtime activation failed",
            ) from exc

        released = working.model_copy(
            update={
                "lease_state": TaskScopedAgentLeaseState.RELEASED,
                "release_runtime_revision_id": request.build.runtime_revision_id,
            },
        )
        if not self._lease_store.compare_and_set(
            working.lease_id,
            expected_state=working.lease_state,
            new_lease=released,
        ):
            current = self._lease_store.get(request.lease_id)
            if (
                current is not None
                and current.lease_state is TaskScopedAgentLeaseState.RELEASED
            ):
                return TaskScopedAgentReleaseResult(
                    outcome=TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED,
                    lease=current,
                    traffic_serving_revision_id=activate_view.traffic_serving_revision_id,
                )
            raise TaskScopedAgentLeaseConflict("lease state changed during release")

        return TaskScopedAgentReleaseResult(
            outcome=TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED,
            lease=released,
            traffic_serving_revision_id=activate_view.traffic_serving_revision_id,
        )


class TaskScopedAgentService:
    """Task-scoped agent ownership facade composing acquisition and release."""

    def __init__(
        self,
        *,
        acquisition: DynamicAgentAcquisitionPort,
        lifecycle: TaskScopedAgentLifecyclePort,
        lease_store: TaskScopedAgentLeaseStore,
    ) -> None:
        self._acquisition_service = TaskScopedAgentAcquisitionService(
            acquisition=acquisition,
            lease_store=lease_store,
        )
        self._release_service = TaskScopedAgentReleaseService(
            lifecycle=lifecycle,
            lease_store=lease_store,
        )

    def acquire(
        self,
        request: TaskScopedAgentAcquisitionRequest,
        *,
        principal: RequestIdentity,
    ) -> TaskScopedAgentAcquisitionResult:
        return self._acquisition_service.acquire(request, principal=principal)

    def release(
        self,
        request: TaskScopedAgentReleaseRequest,
        *,
        principal: RequestIdentity,
    ) -> TaskScopedAgentReleaseResult:
        return self._release_service.release(request, principal=principal)


__all__ = [
    "BindingTaskOrigin",
    "DynamicAgentAcquisitionPort",
    "InMemoryTaskScopedAgentLeaseStore",
    "TaskScopeId",
    "TaskScopedAgentAcquisitionOutcome",
    "TaskScopedAgentAcquisitionRequest",
    "TaskScopedAgentAcquisitionResult",
    "TaskScopedAgentAcquisitionService",
    "TaskScopedAgentContractError",
    "TaskScopedAgentError",
    "TaskScopedAgentLease",
    "TaskScopedAgentLeaseConflict",
    "TaskScopedAgentLeaseId",
    "TaskScopedAgentLeaseNotFound",
    "TaskScopedAgentLeaseState",
    "TaskScopedAgentLeaseStore",
    "TaskScopedAgentLifecyclePort",
    "TaskScopedAgentOwnershipError",
    "TaskScopedAgentReleaseError",
    "TaskScopedAgentReleaseOutcome",
    "TaskScopedAgentReleaseRequest",
    "TaskScopedAgentReleaseResult",
    "TaskScopedAgentReleaseService",
    "TaskScopedAgentService",
    "TaskScopedOwnershipMode",
    "binding_requires_runtime_release",
]
