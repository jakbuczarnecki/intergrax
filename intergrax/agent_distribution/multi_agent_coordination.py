# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical multi-agent coordination contracts above delegated subtasks (NPSC-5A)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Generic, NewType, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.agent_selection import AgentSelectionContext
from intergrax.agent_distribution.delegated_subtasks import (
    ChildBudgetRequest,
    DelegatedSubtaskAcquisitionError,
    DelegatedSubtaskCleanupError,
    DelegatedSubtaskContractError,
    DelegatedSubtaskError,
    DelegatedSubtaskExecutionAndReleaseError,
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskGovernanceRequiresHuman,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskInvocationError,
    DelegatedSubtaskNoEligibleAgent,
    DelegatedSubtaskReleaseError,
    DelegatedSubtaskRequest,
    DelegatedSubtaskResolutionError,
    DelegatedSubtaskResult,
    DelegatedSubtaskService,
    DelegatedSubtaskTaskScopeError,
    DelegatedSubtaskTaskScopeMismatch,
    DelegationId,
    validate_delegation_id,
)
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeed,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopeId,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernedContinuation,
)

_NON_EMPTY = Field(min_length=1)

SCHEMA_COORDINATION_REQUEST_V1: Final = "coordination_request.v1"
SCHEMA_COORDINATION_POLICY_V1: Final = "coordination_policy.v1"

CoordinationId = NewType("CoordinationId", str)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def validate_coordination_id(value: object) -> CoordinationId:
    if type(value) is not str:
        raise TypeError("coordination_id must be str")
    return CoordinationId(_strip_required(value))


def _validate_coordination_id_field(value: object) -> CoordinationId:
    return validate_coordination_id(value)


class CoordinationFailureCode(StrEnum):
    """Bounded semantic categories for coordination failures."""

    INVALID_COORDINATION = "invalid_coordination"
    CAPABILITY_RESOLUTION_FAILED = "capability_resolution_failed"
    NO_ELIGIBLE_SPECIALIST = "no_eligible_specialist"
    ACQUISITION_FAILED = "acquisition_failed"
    CHILD_EXECUTION_FAILED = "child_execution_failed"
    LEASE_RELEASE_FAILED = "lease_release_failed"
    AUTHORITY_SCOPE_MISMATCH = "authority_scope_mismatch"
    GOVERNANCE_DENIED = "governance_denied"
    GOVERNANCE_REQUIRES_HUMAN = "governance_requires_human"


class CoordinationError(AgentDistributionError):
    """Base error for multi-agent coordination boundary violations."""

    failure_code: CoordinationFailureCode

    def __init__(
        self,
        message: str,
        *,
        failure_code: CoordinationFailureCode,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class InvalidCoordinationError(CoordinationError):
    """Malformed coordination request or projection."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.INVALID_COORDINATION,
        )


class CapabilityResolutionFailedError(CoordinationError):
    """Capability resolution or discovery pipeline failure."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.CAPABILITY_RESOLUTION_FAILED,
        )


class NoEligibleSpecialistError(CoordinationError):
    """No discovered candidate satisfies the resolved capability requirement."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.NO_ELIGIBLE_SPECIALIST,
        )


class AcquisitionFailedError(CoordinationError):
    """Task-scoped specialist acquisition failed before child execution."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.ACQUISITION_FAILED,
        )


class ChildExecutionFailedError(CoordinationError):
    """Specialist child execution failed after lease acquisition."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.CHILD_EXECUTION_FAILED,
        )


class LeaseReleaseFailedError(CoordinationError):
    """Lease release failed after delegated execution."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.LEASE_RELEASE_FAILED,
        )


class AuthorityScopeMismatchError(CoordinationError):
    """Task scope or authority validation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.AUTHORITY_SCOPE_MISMATCH,
        )


class GovernanceDeniedError(CoordinationError):
    """Physical delegation governance denied before acquisition."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.GOVERNANCE_DENIED,
        )


class GovernanceRequiresHumanError(CoordinationError):
    """Physical delegation governance requires governed continuation."""

    continuation: PhysicalDelegationGovernedContinuation

    def __init__(
        self,
        message: str,
        *,
        continuation: PhysicalDelegationGovernedContinuation,
    ) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN,
        )
        self.continuation = continuation


class CoordinationPolicy(BaseModel):
    """Typed parent constraints for platform-owned specialist selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COORDINATION_POLICY_V1
    selection_context: AgentSelectionContext = AgentSelectionContext()


class CoordinationRequest(BaseModel):
    """Parent intent for one bounded specialist contribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_COORDINATION_REQUEST_V1
    coordination_id: CoordinationId
    delegation_id: DelegationId
    task_scope_id: TaskScopeId
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    lease_id: TaskScopedAgentLeaseId
    capability_need: AgentDistributionCapabilityNeed
    policy: CoordinationPolicy = CoordinationPolicy()

    @field_validator("coordination_id", mode="before")
    @classmethod
    def _validate_coordination(cls, value: object) -> CoordinationId:
        return _validate_coordination_id_field(value)

    @field_validator("delegation_id", mode="before")
    @classmethod
    def _validate_delegation(cls, value: object) -> DelegationId:
        return validate_delegation_id(value)

    @field_validator("application_id", "application_environment_id")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("lease_id", mode="before")
    @classmethod
    def _validate_lease_id(cls, value: object) -> TaskScopedAgentLeaseId:
        if type(value) is not str:
            raise TypeError("lease_id must be str")
        return TaskScopedAgentLeaseId(_strip_required(value))

    @field_validator("task_scope_id", mode="before")
    @classmethod
    def _validate_task_scope(cls, value: object) -> TaskScopeId:
        from intergrax.contracts.execution_identity import validate_task_id

        return validate_task_id(value)


@dataclass(frozen=True, slots=True)
class CoordinationDelegation(Generic[RequestT]):
    """Typed specialist payload and child authority narrowing for coordination."""

    payload: RequestT
    requested_permission_scopes: tuple[str, ...] | None = None
    requested_budget: ChildBudgetRequest | None = None


@dataclass(frozen=True, slots=True)
class CoordinationResult(Generic[ResultT]):
    """Audit-friendly coordination outcome projected from delegated subtask evidence."""

    coordination_id: CoordinationId
    delegated: DelegatedSubtaskResult[ResultT]

    @property
    def delegation_id(self) -> DelegationId:
        return self.delegated.delegation_id

    @property
    def task_scope_id(self) -> TaskScopeId:
        return self.delegated.task_scope_id

    @property
    def result(self) -> ResultT:
        return self.delegated.result


class CoordinationCleanupError(CoordinationError, Generic[ResultT]):
    """Child execution succeeded but lease cleanup failed."""

    def __init__(
        self,
        message: str,
        *,
        coordination_id: CoordinationId,
        result: ResultT,
        release_cause: BaseException,
    ) -> None:
        super().__init__(
            message,
            failure_code=CoordinationFailureCode.LEASE_RELEASE_FAILED,
        )
        self.coordination_id = coordination_id
        self.result = result
        self.release_cause = release_cause


def build_delegated_subtask_request(
    request: CoordinationRequest,
) -> DelegatedSubtaskRequest:
    """Project coordination intent into the canonical delegated subtask contract."""
    return DelegatedSubtaskRequest(
        delegation_id=request.delegation_id,
        task_scope_id=request.task_scope_id,
        application_id=request.application_id,
        application_environment_id=request.application_environment_id,
        lease_id=request.lease_id,
        capability_need=request.capability_need,
    )


def _map_delegated_subtask_error(exc: DelegatedSubtaskError) -> CoordinationError:
    if isinstance(exc, DelegatedSubtaskContractError):
        return InvalidCoordinationError(str(exc))
    if isinstance(exc, DelegatedSubtaskTaskScopeMismatch):
        return AuthorityScopeMismatchError(str(exc))
    if isinstance(exc, DelegatedSubtaskTaskScopeError):
        return AuthorityScopeMismatchError(str(exc))
    if isinstance(exc, DelegatedSubtaskResolutionError):
        return CapabilityResolutionFailedError(str(exc))
    if isinstance(exc, DelegatedSubtaskNoEligibleAgent):
        return NoEligibleSpecialistError(str(exc))
    if isinstance(exc, DelegatedSubtaskGovernanceDenied):
        return GovernanceDeniedError(str(exc))
    if isinstance(exc, DelegatedSubtaskGovernanceRequiresHuman):
        return GovernanceRequiresHumanError(
            str(exc),
            continuation=exc.continuation,
        )
    if isinstance(exc, DelegatedSubtaskAcquisitionError):
        return AcquisitionFailedError(str(exc))
    if isinstance(exc, DelegatedSubtaskInvocationError):
        return ChildExecutionFailedError(str(exc))
    if isinstance(exc, DelegatedSubtaskReleaseError):
        return LeaseReleaseFailedError(str(exc))
    if isinstance(exc, DelegatedSubtaskExecutionAndReleaseError):
        return ChildExecutionFailedError(str(exc))
    return InvalidCoordinationError(str(exc))


class MultiAgentCoordinationService(Generic[RequestT, ResultT]):
    """Validate coordination intent and delegate through DelegatedSubtaskService."""

    __slots__ = ("_delegated_subtasks",)

    def __init__(
        self,
        *,
        delegated_subtasks: DelegatedSubtaskService[RequestT, ResultT],
    ) -> None:
        self._delegated_subtasks = delegated_subtasks

    async def coordinate(
        self,
        request: CoordinationRequest,
        *,
        delegation: CoordinationDelegation[RequestT],
        principal: RequestIdentity,
    ) -> CoordinationResult[ResultT]:
        delegated_request = build_delegated_subtask_request(request)
        invocation = DelegatedSubtaskInvocation(
            payload=delegation.payload,
            requested_permission_scopes=delegation.requested_permission_scopes,
            requested_budget=delegation.requested_budget,
        )
        try:
            delegated_result = await self._delegated_subtasks.execute(
                delegated_request,
                invocation=invocation,
                principal=principal,
            )
        except DelegatedSubtaskCleanupError as exc:
            raise CoordinationCleanupError(
                "coordination succeeded but lease release failed",
                coordination_id=request.coordination_id,
                result=exc.result,
                release_cause=exc.release_cause,
            ) from exc
        except DelegatedSubtaskError as exc:
            raise _map_delegated_subtask_error(exc) from exc
        return CoordinationResult(
            coordination_id=request.coordination_id,
            delegated=delegated_result,
        )


__all__ = [
    "AcquisitionFailedError",
    "AuthorityScopeMismatchError",
    "CapabilityResolutionFailedError",
    "ChildExecutionFailedError",
    "CoordinationCleanupError",
    "CoordinationDelegation",
    "CoordinationError",
    "CoordinationFailureCode",
    "CoordinationId",
    "CoordinationPolicy",
    "CoordinationRequest",
    "CoordinationResult",
    "GovernanceDeniedError",
    "GovernanceRequiresHumanError",
    "InvalidCoordinationError",
    "LeaseReleaseFailedError",
    "MultiAgentCoordinationService",
    "NoEligibleSpecialistError",
    "build_delegated_subtask_request",
    "validate_coordination_id",
]
