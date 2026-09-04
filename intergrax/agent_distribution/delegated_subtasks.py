# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Delegated subtask discovery and execution orchestration (AC-4 Phase 8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Generic, NewType, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryError,
    AgentDiscoveryRequest,
    AgentDiscoveryStrategy,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_selection import (
    AgentSelectionDecision,
    AgentSelectionStrategy,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import (
    AgentCapabilityRequirement,
    CapabilityMatchResult,
    CapabilityMatcher,
)
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity
from intergrax.agent_distribution.dynamic_acquisition import (
    DynamicAgentAcquisitionResult,
)
from intergrax.agent_distribution.errors import AgentDistributionError
from intergrax.agent_distribution.task_capability_resolution import (
    TaskCapabilityResolutionError,
    TaskCapabilityResolutionRequest,
    TaskCapabilityResolutionResult,
    TaskCapabilityResolver,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentAcquisitionRequest,
    TaskScopedAgentError,
    TaskScopedAgentLease,
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseState,
    TaskScopedAgentReleaseError,
    TaskScopedAgentReleaseRequest,
    TaskScopedAgentReleaseResult,
    TaskScopedAgentService,
    TaskScopeId,
)
from intergrax.contracts.active_execution_task_scope import (
    ActiveExecutionTaskScopePort,
    ActiveExecutionTaskScopeUnavailable,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
)

_NON_EMPTY = Field(min_length=1)

SCHEMA_DELEGATED_SUBTASK_REQUEST_V1: Final = "delegated_subtask_request.v1"

DelegationId = NewType("DelegationId", str)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def validate_delegation_id(value: object) -> DelegationId:
    if type(value) is not str:
        raise TypeError("delegation_id must be str")
    return DelegationId(_strip_required(value))


def _validate_delegation_id_field(value: object) -> DelegationId:
    return validate_delegation_id(value)


class DelegatedSubtaskError(AgentDistributionError):
    """Base error for delegated subtask orchestration."""


class DelegatedSubtaskContractError(DelegatedSubtaskError):
    """Malformed delegated subtask request or result."""


class DelegatedSubtaskTaskScopeError(DelegatedSubtaskError):
    """Task scope authority validation failed."""


class DelegatedSubtaskTaskScopeMismatch(DelegatedSubtaskTaskScopeError):
    """Caller task_scope_id does not match canonical active execution task scope."""


class DelegatedSubtaskResolutionError(DelegatedSubtaskError):
    """Capability resolution or discovery pipeline failure."""


class DelegatedSubtaskNoEligibleAgent(DelegatedSubtaskError):
    """No discovered candidate satisfies the resolved capability requirement."""


class DelegatedSubtaskAcquisitionError(DelegatedSubtaskError):
    """Task-scoped specialist acquisition failed before child execution."""


class DelegatedSubtaskInvocationError(DelegatedSubtaskError):
    """Specialist child execution failed after lease acquisition."""


class DelegatedSubtaskReleaseError(DelegatedSubtaskError):
    """Lease release failed after successful child execution."""


class DelegatedSubtaskExecutionAndReleaseError(DelegatedSubtaskError):
    """Child execution and lease release both failed."""

    def __init__(
        self,
        message: str,
        *,
        execution_cause: BaseException,
        release_cause: BaseException,
    ) -> None:
        super().__init__(message)
        self.execution_cause = execution_cause
        self.release_cause = release_cause


class DelegatedSubtaskCleanupError(DelegatedSubtaskError, Generic[ResultT]):
    """Child execution succeeded but lease cleanup failed."""

    def __init__(
        self,
        message: str,
        *,
        result: ResultT,
        release_cause: BaseException,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.release_cause = release_cause


class DelegatedSubtaskRequest(BaseModel):
    """Functional specialist need inside an active parent execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_DELEGATED_SUBTASK_REQUEST_V1
    delegation_id: DelegationId
    task_scope_id: TaskScopeId
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    lease_id: TaskScopedAgentLeaseId
    capability_resolution_request: TaskCapabilityResolutionRequest

    @field_validator("delegation_id", mode="before")
    @classmethod
    def _validate_delegation(cls, value: object) -> DelegationId:
        return _validate_delegation_id_field(value)

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
class DelegatedSubtaskInvocation(Generic[RequestT]):
    """Typed specialist payload executed through canonical child execution."""

    payload: RequestT
    requested_permission_scopes: tuple[str, ...] | None = None
    requested_budget: ChildBudgetRequest | None = None


@dataclass(frozen=True, slots=True)
class DelegatedSubtaskLifecyclePlan:
    """Pure acquisition plan — no lifecycle I/O."""

    acquisition_request: TaskScopedAgentAcquisitionRequest


@dataclass(frozen=True, slots=True)
class DelegatedSubtaskReleaseContext:
    """Current lease authority used to build a fresh release request."""

    lease: TaskScopedAgentLease
    task_scope_id: TaskScopeId
    application_id: str
    application_environment_id: str
    selected_identity: AgentDiscoveryCandidateIdentity


@dataclass(frozen=True, slots=True)
class DelegatedSubtaskResult(Generic[ResultT]):
    """Audit-friendly delegated subtask outcome."""

    delegation_id: DelegationId
    task_scope_id: TaskScopeId
    capability_resolution: TaskCapabilityResolutionResult
    capability_requirement: AgentCapabilityRequirement
    match_results: tuple[CapabilityMatchResult, ...]
    selection_decision: AgentSelectionDecision
    selected_identity: AgentDiscoveryCandidateIdentity
    lease_id: TaskScopedAgentLeaseId
    application_binding_id: str
    acquisition_result: DynamicAgentAcquisitionResult
    release_result: TaskScopedAgentReleaseResult
    result: ResultT


class DelegatedSubtaskAcquisitionPlanFactory(Protocol):
    """Build canonical task-scoped acquisition requests from selected identity."""

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id: TaskScopeId,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity: AgentDiscoveryCandidateIdentity,
    ) -> DelegatedSubtaskLifecyclePlan: ...


class DelegatedSubtaskReleasePlanFactory(Protocol):
    """Build release requests from current lease state immediately before release."""

    def build_release_request(
        self,
        *,
        context: DelegatedSubtaskReleaseContext,
    ) -> TaskScopedAgentReleaseRequest: ...


class DelegatedSubtaskDelegate(Protocol[RequestT, ResultT]):
    """Specialist executor invoked through canonical child execution."""

    async def execute(self, request: RequestT) -> ResultT: ...


class ChildBudgetRequest(Protocol):
    """Minimal child budget contract accepted by runtime adapters."""

    def validate(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DelegatedChildExecutionOptions:
    requested_permission_scopes: tuple[str, ...] | None = None
    requested_budget: ChildBudgetRequest | None = None


class ChildExecutionPort(Protocol[RequestT, ResultT]):
    """Canonical child execution boundary — runtime adapter required."""

    async def execute_child(
        self,
        *,
        request: RequestT,
        delegate: DelegatedSubtaskDelegate[RequestT, ResultT],
        options: DelegatedChildExecutionOptions | None = None,
    ) -> ResultT: ...


class SpecialistInvocationPort(Protocol[RequestT, ResultT]):
    """Resolve an active task-scoped lease into an executable specialist delegate."""

    def resolve_delegate(
        self,
        *,
        lease: TaskScopedAgentLease,
        acquisition_result: DynamicAgentAcquisitionResult,
    ) -> DelegatedSubtaskDelegate[RequestT, ResultT]: ...


def _eligible_matches(
    *,
    match_results: tuple[CapabilityMatchResult, ...],
) -> tuple[CapabilityMatchResult, ...]:
    return tuple(match for match in match_results if match.eligible)


def _resolve_canonical_task_scope(
    task_scope_authority: ActiveExecutionTaskScopePort,
) -> TaskScopeId:
    run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    try:
        return task_scope_authority.resolve_current_task_scope(
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
    except ActiveExecutionTaskScopeUnavailable as exc:
        raise DelegatedSubtaskTaskScopeError(
            "canonical task scope unavailable for active execution",
        ) from exc


def _validate_acquisition_plan(
    *,
    request: DelegatedSubtaskRequest,
    canonical_task_scope: TaskScopeId,
    selected_identity: AgentDiscoveryCandidateIdentity,
    lifecycle_plan: DelegatedSubtaskLifecyclePlan,
) -> None:
    acquisition_request = lifecycle_plan.acquisition_request
    dynamic_request = acquisition_request.acquisition_request
    if acquisition_request.task_scope_id != canonical_task_scope:
        raise DelegatedSubtaskContractError(
            "acquisition plan task_scope_id does not match canonical task scope",
        )
    if acquisition_request.lease_id != request.lease_id:
        raise DelegatedSubtaskContractError(
            "acquisition plan lease_id does not match delegated request",
        )
    if dynamic_request.application_id != request.application_id:
        raise DelegatedSubtaskContractError(
            "acquisition plan application_id does not match delegated request",
        )
    if dynamic_request.application_environment_id != request.application_environment_id:
        raise DelegatedSubtaskContractError(
            "acquisition plan application_environment_id does not match delegated request",
        )
    if dynamic_request.selected_identity != selected_identity:
        raise DelegatedSubtaskContractError(
            "acquisition plan selected_identity does not match selection decision",
        )


def _validate_release_request(
    *,
    request: DelegatedSubtaskRequest,
    canonical_task_scope: TaskScopeId,
    lease: TaskScopedAgentLease,
    release_request: TaskScopedAgentReleaseRequest,
) -> None:
    if release_request.lease_id != lease.lease_id:
        raise DelegatedSubtaskContractError(
            "release plan lease_id does not match acquired lease",
        )
    if release_request.task_scope_id != canonical_task_scope:
        raise DelegatedSubtaskContractError(
            "release plan task_scope_id does not match canonical task scope",
        )
    if release_request.application_id != request.application_id:
        raise DelegatedSubtaskContractError(
            "release plan application_id does not match delegated request",
        )
    if release_request.application_environment_id != request.application_environment_id:
        raise DelegatedSubtaskContractError(
            "release plan application_environment_id does not match delegated request",
        )


class DelegatedSubtaskService(Generic[RequestT, ResultT]):
    """Thin orchestration over AC-4 discovery, Phase 7 leases, and child execution."""

    def __init__(
        self,
        *,
        capability_resolver: TaskCapabilityResolver,
        discovery: AgentDiscoveryStrategy,
        matcher: CapabilityMatcher,
        selector: AgentSelectionStrategy,
        task_scoped_agents: TaskScopedAgentService,
        task_scope_authority: ActiveExecutionTaskScopePort,
        acquisition_plan_factory: DelegatedSubtaskAcquisitionPlanFactory,
        release_plan_factory: DelegatedSubtaskReleasePlanFactory,
        specialist_invocation: SpecialistInvocationPort[RequestT, ResultT],
        child_execution: ChildExecutionPort[RequestT, ResultT],
    ) -> None:
        self._capability_resolver = capability_resolver
        self._discovery = discovery
        self._matcher = matcher
        self._selector = selector
        self._task_scoped_agents = task_scoped_agents
        self._task_scope_authority = task_scope_authority
        self._acquisition_plan_factory = acquisition_plan_factory
        self._release_plan_factory = release_plan_factory
        self._specialist_invocation = specialist_invocation
        self._child_execution = child_execution

    async def execute(
        self,
        request: DelegatedSubtaskRequest,
        *,
        invocation: DelegatedSubtaskInvocation[RequestT],
        principal: RequestIdentity,
    ) -> DelegatedSubtaskResult[ResultT]:
        canonical_task_scope = _resolve_canonical_task_scope(self._task_scope_authority)
        if request.task_scope_id != canonical_task_scope:
            raise DelegatedSubtaskTaskScopeMismatch(
                "delegated subtask task_scope_id does not match canonical active task scope",
            )

        try:
            capability_resolution = self._capability_resolver.resolve(
                request.capability_resolution_request,
            )
        except TaskCapabilityResolutionError as exc:
            raise DelegatedSubtaskResolutionError(
                "task capability resolution failed",
            ) from exc

        requirement = capability_resolution.capability_requirement
        try:
            discovery_result = self._discovery.discover(
                AgentDiscoveryRequest(requirement=requirement),
            )
        except AgentDiscoveryError as exc:
            raise DelegatedSubtaskResolutionError(
                "agent discovery failed",
            ) from exc

        match_results = self._matcher.find_matches(
            requirement=requirement,
            candidates=tuple(
                project_to_capability_candidate(candidate)
                for candidate in discovery_result.candidates
            ),
        )
        eligible_matches = _eligible_matches(match_results=match_results)
        selection_decision = self._selector.select(
            build_agent_selection_request(
                requirement=requirement,
                eligible_matches=eligible_matches,
            ),
        )
        if selection_decision.outcome is not SelectionOutcome.SELECTED:
            raise DelegatedSubtaskNoEligibleAgent(
                "no eligible specialist candidate for delegated subtask",
            )
        selected_identity = require_selected_identity(selection_decision)

        lifecycle_plan = self._acquisition_plan_factory.build_acquisition_plan(
            delegation_id=request.delegation_id,
            task_scope_id=canonical_task_scope,
            application_id=request.application_id,
            application_environment_id=request.application_environment_id,
            lease_id=request.lease_id,
            selected_identity=selected_identity,
        )
        _validate_acquisition_plan(
            request=request,
            canonical_task_scope=canonical_task_scope,
            selected_identity=selected_identity,
            lifecycle_plan=lifecycle_plan,
        )
        try:
            acquisition = self._task_scoped_agents.acquire(
                lifecycle_plan.acquisition_request,
                principal=principal,
            )
        except TaskScopedAgentError as exc:
            raise DelegatedSubtaskAcquisitionError(
                "task-scoped specialist acquisition failed",
            ) from exc

        if acquisition.lease.lease_state is not TaskScopedAgentLeaseState.ACTIVE:
            raise DelegatedSubtaskAcquisitionError(
                "task-scoped lease is not active before specialist execution",
            )

        delegate = self._specialist_invocation.resolve_delegate(
            lease=acquisition.lease,
            acquisition_result=acquisition.acquisition_result,
        )
        child_options = DelegatedChildExecutionOptions(
            requested_permission_scopes=invocation.requested_permission_scopes,
            requested_budget=invocation.requested_budget,
        )
        try:
            specialist_result = await self._child_execution.execute_child(
                request=invocation.payload,
                delegate=delegate,
                options=child_options,
            )
        except BaseException as execution_exc:
            release_exc = self._attempt_release(
                request=request,
                canonical_task_scope=canonical_task_scope,
                lease=acquisition.lease,
                selected_identity=selected_identity,
                principal=principal,
            )
            if release_exc is not None:
                raise DelegatedSubtaskExecutionAndReleaseError(
                    "delegated subtask execution and lease release failed",
                    execution_cause=execution_exc,
                    release_cause=release_exc,
                ) from execution_exc
            if isinstance(execution_exc, Exception):
                raise DelegatedSubtaskInvocationError(
                    "specialist child execution failed",
                ) from execution_exc
            raise

        try:
            release_result = self._release(
                request=request,
                canonical_task_scope=canonical_task_scope,
                lease=acquisition.lease,
                selected_identity=selected_identity,
                principal=principal,
            )
        except DelegatedSubtaskReleaseError as release_exc:
            raise DelegatedSubtaskCleanupError(
                "delegated subtask succeeded but lease release failed",
                result=specialist_result,
                release_cause=release_exc,
            ) from release_exc
        return DelegatedSubtaskResult(
            delegation_id=request.delegation_id,
            task_scope_id=request.task_scope_id,
            capability_resolution=capability_resolution,
            capability_requirement=requirement,
            match_results=match_results,
            selection_decision=selection_decision,
            selected_identity=selected_identity,
            lease_id=request.lease_id,
            application_binding_id=acquisition.lease.application_binding_id,
            acquisition_result=acquisition.acquisition_result,
            release_result=release_result,
            result=specialist_result,
        )

    def _attempt_release(
        self,
        *,
        request: DelegatedSubtaskRequest,
        canonical_task_scope: TaskScopeId,
        lease: TaskScopedAgentLease,
        selected_identity: AgentDiscoveryCandidateIdentity,
        principal: RequestIdentity,
    ) -> BaseException | None:
        try:
            self._release(
                request=request,
                canonical_task_scope=canonical_task_scope,
                lease=lease,
                selected_identity=selected_identity,
                principal=principal,
            )
        except BaseException as exc:
            return exc
        return None

    def _release(
        self,
        *,
        request: DelegatedSubtaskRequest,
        canonical_task_scope: TaskScopeId,
        lease: TaskScopedAgentLease,
        selected_identity: AgentDiscoveryCandidateIdentity,
        principal: RequestIdentity,
    ) -> TaskScopedAgentReleaseResult:
        release_request = self._release_plan_factory.build_release_request(
            context=DelegatedSubtaskReleaseContext(
                lease=lease,
                task_scope_id=canonical_task_scope,
                application_id=request.application_id,
                application_environment_id=request.application_environment_id,
                selected_identity=selected_identity,
            ),
        )
        _validate_release_request(
            request=request,
            canonical_task_scope=canonical_task_scope,
            lease=lease,
            release_request=release_request,
        )
        try:
            return self._task_scoped_agents.release(
                release_request,
                principal=principal,
            )
        except TaskScopedAgentReleaseError as exc:
            raise DelegatedSubtaskReleaseError(
                "task-scoped specialist release failed",
            ) from exc


__all__ = [
    "ChildBudgetRequest",
    "ChildExecutionPort",
    "DelegatedChildExecutionOptions",
    "DelegatedSubtaskAcquisitionPlanFactory",
    "DelegatedSubtaskCleanupError",
    "DelegatedSubtaskContractError",
    "DelegatedSubtaskDelegate",
    "DelegatedSubtaskError",
    "DelegatedSubtaskExecutionAndReleaseError",
    "DelegatedSubtaskInvocation",
    "DelegatedSubtaskInvocationError",
    "DelegatedSubtaskLifecyclePlan",
    "DelegatedSubtaskNoEligibleAgent",
    "DelegatedSubtaskAcquisitionError",
    "DelegatedSubtaskReleaseContext",
    "DelegatedSubtaskReleaseError",
    "DelegatedSubtaskReleasePlanFactory",
    "DelegatedSubtaskRequest",
    "DelegatedSubtaskTaskScopeError",
    "DelegatedSubtaskTaskScopeMismatch",
    "DelegatedSubtaskResult",
    "DelegatedSubtaskService",
    "DelegationId",
    "SpecialistInvocationPort",
    "validate_delegation_id",
]
