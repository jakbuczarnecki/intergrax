# © Artur Czarnecki. All rights reserved.

"""Authoritative Shared Work mutation service (COLLAB-WORK-2C).

Owns WorkItem and Assignment business mutations with fresh MP-1 authority
enforcement, COLLAB-WORK-2A lifecycle validation, and COLLAB-WORK-2B CAS
persistence. Does not infer WorkItem state from Nexus execution.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.repository import (
    AssignmentNotFound,
    AssignmentRepository,
    AssignmentRevisionConflict,
    AssignmentScopeKey,
    CreateAssignmentCommand,
    CreateWorkItemCommand,
    UpdateAssignmentCommand,
    UpdateWorkItemCommand,
    WorkItemNotFound,
    WorkItemRepository,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
)
from intergrax.contracts.collaborative_work import (
    Assignment,
    AssignmentTransitionRequest,
    AuthorityDelegation,
    CollaborativeWorkAuthorizationDenied,
    CollaborativeWorkEnforcementRequest,
    CollaborativeWorkEnforcementResult,
    CollaborativeWorkLifecycleError,
    CreateAssignmentRequest,
    CreateWorkItemRequest,
    MembershipResolutionMode,
    TransitionAssignmentRequest,
    TransitionWorkItemRequest,
    WorkItem,
    WorkItemTransitionRequest,
    WorkspaceMembership,
    apply_assignment_transition,
    apply_work_item_transition,
    validate_assignment_state_transition,
    validate_work_item_state_transition,
    work_item_resource_scope,
)
from intergrax.contracts.runtime_policy import PolicyAction

TRUSTED_OPERATION_WORK_ITEM_CREATE: Final = "collaborative_work.work_item.create"
TRUSTED_OPERATION_WORK_ITEM_TRANSITION: Final = "collaborative_work.work_item.transition"
TRUSTED_OPERATION_ASSIGNMENT_CREATE: Final = "collaborative_work.assignment.create"
TRUSTED_OPERATION_ASSIGNMENT_TRANSITION: Final = "collaborative_work.assignment.transition"


@runtime_checkable
class _AuthorityContextRequest(Protocol):
    tenant_id: str
    workspace_id: str
    acting_principal_id: str
    delegator_principal_id: str | None
    membership: WorkspaceMembership | None
    membership_resolution_mode: MembershipResolutionMode
    delegation: AuthorityDelegation | None


class CollaborativeWorkService:
    """Authoritative Shared Work mutation boundary for WorkItem and Assignment."""

    def __init__(
        self,
        *,
        work_item_repository: WorkItemRepository,
        assignment_repository: AssignmentRepository,
        enforcement_gate: CollaborativeWorkEnforcementGate,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._work_item_repository = work_item_repository
        self._assignment_repository = assignment_repository
        self._enforcement_gate = enforcement_gate
        self._clock = clock or (lambda: datetime.now(UTC))

    def create_work_item(self, request: CreateWorkItemRequest) -> WorkItem:
        resource_scope = work_item_resource_scope(work_item_id=request.work_item_id)
        self._require_allow(
            operation_id=TRUSTED_OPERATION_WORK_ITEM_CREATE,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._clock()
        return self._work_item_repository.create(
            CreateWorkItemCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                work_item_id=request.work_item_id,
                created_by_principal_id=request.acting_principal_id,
                created_at=now,
                updated_at=now,
                title=request.title,
                description=request.description,
                idempotency_key=request.idempotency_key,
            ),
        )

    def transition_work_item(self, request: TransitionWorkItemRequest) -> WorkItem:
        work_item = self._load_work_item(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
        )
        self._assert_work_item_revision(work_item, expected_revision=request.expected_revision)
        self._validate_work_item_transition(work_item=work_item, request=request)
        resource_scope = work_item_resource_scope(work_item_id=work_item.work_item_id)
        self._require_allow(
            operation_id=TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
            request=request,
            resource_scope=resource_scope,
        )
        updated_at = self._clock()
        transition = self._work_item_transition_request(request)
        next_state = apply_work_item_transition(
            work_item,
            transition,
            updated_at=updated_at,
        )
        return self._work_item_repository.update(
            UpdateWorkItemCommand(
                scope=WorkItemScopeKey(
                    tenant_id=work_item.tenant_id,
                    workspace_id=work_item.workspace_id,
                    work_item_id=work_item.work_item_id,
                ),
                expected_revision=work_item.revision,
                state=next_state.state,
                updated_at=next_state.updated_at,
                title=work_item.title,
                description=work_item.description,
            ),
        )

    def create_assignment(self, request: CreateAssignmentRequest) -> Assignment:
        work_item = self._load_work_item(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
        )
        resource_scope = work_item_resource_scope(work_item_id=work_item.work_item_id)
        self._require_allow(
            operation_id=TRUSTED_OPERATION_ASSIGNMENT_CREATE,
            request=request,
            resource_scope=resource_scope,
        )
        now = self._clock()
        return self._assignment_repository.create(
            CreateAssignmentCommand(
                tenant_id=request.tenant_id,
                workspace_id=request.workspace_id,
                assignment_id=request.assignment_id,
                work_item_id=request.work_item_id,
                principal_id=request.principal_id,
                created_by_principal_id=request.acting_principal_id,
                created_at=now,
                updated_at=now,
                idempotency_key=request.idempotency_key,
            ),
        )

    def transition_assignment(self, request: TransitionAssignmentRequest) -> Assignment:
        assignment = self._load_assignment(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            assignment_id=request.assignment_id,
        )
        self._assert_assignment_revision(assignment, expected_revision=request.expected_revision)
        self._validate_assignment_transition(assignment=assignment, request=request)
        resource_scope = work_item_resource_scope(work_item_id=assignment.work_item_id)
        self._require_allow(
            operation_id=TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
            request=request,
            resource_scope=resource_scope,
        )
        updated_at = self._clock()
        transition = self._assignment_transition_request(request)
        next_state = apply_assignment_transition(
            assignment,
            transition,
            updated_at=updated_at,
        )
        return self._assignment_repository.update(
            UpdateAssignmentCommand(
                scope=AssignmentScopeKey(
                    tenant_id=assignment.tenant_id,
                    workspace_id=assignment.workspace_id,
                    assignment_id=assignment.assignment_id,
                ),
                expected_revision=assignment.revision,
                state=next_state.state,
                updated_at=next_state.updated_at,
            ),
        )

    def _load_work_item(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> WorkItem:
        work_item = self._work_item_repository.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_item_id=work_item_id,
        )
        if work_item is None:
            raise WorkItemNotFound("work item was not found")
        return work_item

    def _load_assignment(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        assignment_id: str,
    ) -> Assignment:
        assignment = self._assignment_repository.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            assignment_id=assignment_id,
        )
        if assignment is None:
            raise AssignmentNotFound("assignment was not found")
        return assignment

    @staticmethod
    def _assert_work_item_revision(work_item: WorkItem, *, expected_revision: int) -> None:
        if expected_revision != work_item.revision:
            raise WorkItemRevisionConflict("work item revision conflict")

    @staticmethod
    def _assert_assignment_revision(assignment: Assignment, *, expected_revision: int) -> None:
        if expected_revision != assignment.revision:
            raise AssignmentRevisionConflict("assignment revision conflict")

    @staticmethod
    def _validate_work_item_transition(
        *,
        work_item: WorkItem,
        request: TransitionWorkItemRequest,
    ) -> None:
        if request.tenant_id != work_item.tenant_id:
            raise CollaborativeWorkLifecycleError("transition tenant_id must match WorkItem tenant_id")
        if request.workspace_id != work_item.workspace_id:
            raise CollaborativeWorkLifecycleError(
                "transition workspace_id must match WorkItem workspace_id",
            )
        if request.work_item_id != work_item.work_item_id:
            raise CollaborativeWorkLifecycleError(
                "transition work_item_id must match WorkItem work_item_id",
            )
        validate_work_item_state_transition(
            from_state=work_item.state,
            to_state=request.target_state,
        )

    @staticmethod
    def _validate_assignment_transition(
        *,
        assignment: Assignment,
        request: TransitionAssignmentRequest,
    ) -> None:
        if request.tenant_id != assignment.tenant_id:
            raise CollaborativeWorkLifecycleError(
                "transition tenant_id must match Assignment tenant_id",
            )
        if request.workspace_id != assignment.workspace_id:
            raise CollaborativeWorkLifecycleError(
                "transition workspace_id must match Assignment workspace_id",
            )
        if request.assignment_id != assignment.assignment_id:
            raise CollaborativeWorkLifecycleError(
                "transition assignment_id must match Assignment assignment_id",
            )
        if request.work_item_id != assignment.work_item_id:
            raise CollaborativeWorkLifecycleError(
                "transition work_item_id must match Assignment work_item_id",
            )
        validate_assignment_state_transition(
            from_state=assignment.state,
            to_state=request.target_state,
        )

    def _require_allow(
        self,
        *,
        operation_id: str,
        request: _AuthorityContextRequest,
        resource_scope: str,
    ) -> CollaborativeWorkEnforcementResult:
        enforcement_request = CollaborativeWorkEnforcementRequest(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            operation_id=operation_id,
            acting_principal_id=request.acting_principal_id,
            delegator_principal_id=request.delegator_principal_id,
            resource_scope=resource_scope,
            membership=request.membership,
            membership_resolution_mode=request.membership_resolution_mode,
            delegation=request.delegation,
        )
        result = self._enforcement_gate.evaluate(enforcement_request)
        if result.composition.decision.action is not PolicyAction.ALLOW:
            raise CollaborativeWorkAuthorizationDenied(enforcement_result=result)
        return result

    @staticmethod
    def _work_item_transition_request(
        request: TransitionWorkItemRequest,
    ) -> WorkItemTransitionRequest:
        return WorkItemTransitionRequest(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            work_item_id=request.work_item_id,
            expected_revision=request.expected_revision,
            target_state=request.target_state,
            acting_principal_id=request.acting_principal_id,
            idempotency_key=request.idempotency_key,
        )

    @staticmethod
    def _assignment_transition_request(
        request: TransitionAssignmentRequest,
    ) -> AssignmentTransitionRequest:
        return AssignmentTransitionRequest(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            assignment_id=request.assignment_id,
            work_item_id=request.work_item_id,
            expected_revision=request.expected_revision,
            target_state=request.target_state,
            acting_principal_id=request.acting_principal_id,
            idempotency_key=request.idempotency_key,
        )
