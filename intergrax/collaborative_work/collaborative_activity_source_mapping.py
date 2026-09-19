# © Artur Czarnecki. All rights reserved.

"""MP-6F — pure typed Collaborative Work / ContextView → activity publication mapping."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.collaborative_work.collaborative_activity_source_ports import (
    PublishedWorkArtifactPublicationResult,
)
from intergrax.contracts.collaborative_activity import (
    ActivityIdempotencyKey,
    ArtifactVersionActivityProvenanceRef,
    AssignmentActivityTargetRef,
    CollaborativeActivityActorRef,
    CollaborativeActivityBuiltinSource,
    CollaborativeActivityBuiltinType,
    CollaborativeActivityCorrelation,
    CollaborativeActivityDurabilityClass,
    CollaborativeActivityOutcome,
    CollaborativeActivityOutcomeStatus,
    CollaborativeActivityPublication,
    CollaborativeActivityProvenanceRef,
    CollaborativeActivityScope,
    CollaborativeActivitySourceId,
    CollaborativeActivityTypeId,
    CollaborativeDecisionBindingActivityTargetRef,
    ContextViewActivityProvenanceRef,
    ContextViewActivityTargetRef,
    DecisionActivityProvenanceRef,
    ExecutionActivityProvenanceRef,
    WorkArtifactActivityTargetRef,
    WorkArtifactVersionActivityTargetRef,
    WorkItemActivityTargetRef,
)
from intergrax.contracts.collaborative_decision_binding import CollaborativeDecisionBinding
from intergrax.contracts.collaborative_work import (
    Assignment,
    AuthorityDelegation,
    CreateAssignmentRequest,
    CreateWorkArtifactFromExecutionRequest,
    CreateWorkArtifactRequest,
    CreateWorkItemRequest,
    PrincipalKind,
    PublishWorkArtifactVersionFromExecutionRequest,
    PublishWorkArtifactVersionRequest,
    TransitionAssignmentRequest,
    TransitionWorkItemRequest,
    WorkArtifactVersionRef,
    WorkItem,
)
from intergrax.contracts.context_view import ContextView
from intergrax.contracts.context_view_composition import ContextViewCompositionRequest
from intergrax.contracts.collaborative_decision_binding import CreateCollaborativeDecisionBindingRequest
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef


class CollaborativeActivitySourceMappingError(Exception):
    """Required source-owned truth missing or inconsistent for activity translation."""


@runtime_checkable
class CollaborativeActivityActorPrincipalKindResolver(Protocol):
    """Resolve ``PrincipalKind`` for an acting principal — wired at composition root."""

    def resolve_principal_kind(
        self,
        *,
        tenant_id: str,
        acting_principal_id: str,
    ) -> PrincipalKind:
        ...


@runtime_checkable
class _CollaborativeWorkActingPrincipalRequest(Protocol):
    tenant_id: str
    acting_principal_id: str
    delegator_principal_id: str | None
    delegation: AuthorityDelegation | None


def _activity_idempotency_key(
    *,
    tenant_id: str,
    workspace_id: str,
    source: CollaborativeActivitySourceId,
    source_stable_id: str,
    activity_type: CollaborativeActivityTypeId,
) -> ActivityIdempotencyKey:
    return ActivityIdempotencyKey(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source=source,
        source_stable_id=source_stable_id,
        activity_type=activity_type,
    )


def _scope_from_collaborative_work(
    *,
    tenant_id: str,
    workspace_id: str,
    work_item_id: str,
) -> CollaborativeActivityScope:
    return CollaborativeActivityScope(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
    )


def actor_ref_from_collaborative_work_request(
    request: _CollaborativeWorkActingPrincipalRequest,
    *,
    principal_kind: PrincipalKind,
) -> CollaborativeActivityActorRef:
    delegation_id: str | None = None
    delegator_principal_id: str | None = None
    if request.delegation is not None:
        delegation_id = request.delegation.delegation_id
        delegator_principal_id = request.delegation.delegator_principal_id
    elif request.delegator_principal_id is not None:
        raise CollaborativeActivitySourceMappingError(
            "delegator_principal_id without canonical delegation object is insufficient for actor mapping",
        )
    return CollaborativeActivityActorRef(
        tenant_id=request.tenant_id,
        principal_id=request.acting_principal_id,
        principal_kind=principal_kind,
        delegation_id=delegation_id,
        delegator_principal_id=delegator_principal_id,
    )


def _resolve_actor_from_request(
    request: _CollaborativeWorkActingPrincipalRequest,
    *,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
) -> CollaborativeActivityActorRef:
    principal_kind = principal_kind_resolver.resolve_principal_kind(
        tenant_id=request.tenant_id,
        acting_principal_id=request.acting_principal_id,
    )
    return actor_ref_from_collaborative_work_request(request, principal_kind=principal_kind)


def _succeeded_outcome() -> CollaborativeActivityOutcome:
    return CollaborativeActivityOutcome(status=CollaborativeActivityOutcomeStatus.SUCCEEDED)


def _work_artifact_version_ref(version: PublishedWorkArtifactPublicationResult) -> WorkArtifactVersionRef:
    v = version.version
    return WorkArtifactVersionRef(
        tenant_id=v.tenant_id,
        workspace_id=v.workspace_id,
        work_item_id=v.work_item_id,
        work_artifact_id=v.work_artifact_id,
        work_artifact_version_id=v.work_artifact_version_id,
    )


def _execution_provenance_refs(
    execution: ExecutionProvenanceRef | None,
) -> tuple[ExecutionActivityProvenanceRef, ...]:
    if execution is None:
        return ()
    return (ExecutionActivityProvenanceRef(execution=execution),)


def _correlation_from_operation_id(operation_id: str | None) -> CollaborativeActivityCorrelation | None:
    if operation_id is None:
        return None
    normalized = operation_id.strip()
    if not normalized:
        return None
    return CollaborativeActivityCorrelation(operation_id=normalized)


def map_work_item_created_publication(
    *,
    request: CreateWorkItemRequest,
    work_item: WorkItem,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    if work_item.tenant_id != request.tenant_id or work_item.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("work item scope must match create request")
    if work_item.work_item_id != request.work_item_id:
        raise CollaborativeActivitySourceMappingError("work item identity must match create request")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_CREATED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=work_item.tenant_id,
            workspace_id=work_item.workspace_id,
            work_item_id=work_item.work_item_id,
        ),
        target=WorkItemActivityTargetRef(work_item_id=work_item.work_item_id),
        outcome=_succeeded_outcome(),
        occurred_at=work_item.created_at,
        requested_durability_class=requested_durability_class,
    )


def map_work_item_state_changed_publication(
    *,
    request: TransitionWorkItemRequest,
    work_item: WorkItem,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    if work_item.tenant_id != request.tenant_id or work_item.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("work item scope must match transition request")
    if work_item.work_item_id != request.work_item_id:
        raise CollaborativeActivitySourceMappingError("work item identity must match transition request")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.WORK_ITEM_STATE_CHANGED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=work_item.tenant_id,
            workspace_id=work_item.workspace_id,
            work_item_id=work_item.work_item_id,
        ),
        target=WorkItemActivityTargetRef(work_item_id=work_item.work_item_id),
        outcome=_succeeded_outcome(),
        occurred_at=work_item.updated_at,
        requested_durability_class=requested_durability_class,
    )


def map_assignment_created_publication(
    *,
    request: CreateAssignmentRequest,
    assignment: Assignment,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    if assignment.tenant_id != request.tenant_id or assignment.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("assignment scope must match create request")
    if assignment.assignment_id != request.assignment_id:
        raise CollaborativeActivitySourceMappingError("assignment identity must match create request")
    if assignment.created_at is None:
        raise CollaborativeActivitySourceMappingError("assignment created_at is required for occurred_at")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.ASSIGNMENT_CREATED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=assignment.tenant_id,
            workspace_id=assignment.workspace_id,
            work_item_id=assignment.work_item_id,
        ),
        target=AssignmentActivityTargetRef(
            assignment_id=assignment.assignment_id,
            work_item_id=assignment.work_item_id,
        ),
        outcome=_succeeded_outcome(),
        occurred_at=assignment.created_at,
        requested_durability_class=requested_durability_class,
    )


def map_assignment_state_changed_publication(
    *,
    request: TransitionAssignmentRequest,
    assignment: Assignment,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    if assignment.tenant_id != request.tenant_id or assignment.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("assignment scope must match transition request")
    if assignment.assignment_id != request.assignment_id:
        raise CollaborativeActivitySourceMappingError("assignment identity must match transition request")
    if assignment.updated_at is None:
        raise CollaborativeActivitySourceMappingError("assignment updated_at is required for occurred_at")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.ASSIGNMENT_STATE_CHANGED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=assignment.tenant_id,
            workspace_id=assignment.workspace_id,
            work_item_id=assignment.work_item_id,
        ),
        target=AssignmentActivityTargetRef(
            assignment_id=assignment.assignment_id,
            work_item_id=assignment.work_item_id,
        ),
        outcome=_succeeded_outcome(),
        occurred_at=assignment.updated_at,
        requested_durability_class=requested_durability_class,
    )


def map_work_artifact_created_publication(
    *,
    request: CreateWorkArtifactRequest | CreateWorkArtifactFromExecutionRequest,
    published: PublishedWorkArtifactPublicationResult,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    artifact = published.artifact
    version = published.version
    if artifact.tenant_id != request.tenant_id or artifact.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("artifact scope must match create request")
    if artifact.work_item_id != request.work_item_id:
        raise CollaborativeActivitySourceMappingError("artifact work_item_id must match create request")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    version_ref = _work_artifact_version_ref(published)
    execution = version.execution
    if isinstance(request, CreateWorkArtifactFromExecutionRequest):
        execution = request.execution
    provenance: tuple[CollaborativeActivityProvenanceRef, ...] = (
        ArtifactVersionActivityProvenanceRef(version_ref=version_ref),
    ) + _execution_provenance_refs(execution)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.WORK_ARTIFACT_CREATED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=artifact.tenant_id,
            workspace_id=artifact.workspace_id,
            work_item_id=artifact.work_item_id,
        ),
        target=WorkArtifactActivityTargetRef(
            work_artifact_id=artifact.work_artifact_id,
            work_item_id=artifact.work_item_id,
        ),
        outcome=_succeeded_outcome(),
        occurred_at=version.published_at,
        provenance_refs=provenance,
        requested_durability_class=requested_durability_class,
    )


def map_work_artifact_version_published_publication(
    *,
    request: PublishWorkArtifactVersionRequest | PublishWorkArtifactVersionFromExecutionRequest,
    published: PublishedWorkArtifactPublicationResult,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.COLLABORATIVE
    ),
) -> CollaborativeActivityPublication:
    version = published.version
    if version.tenant_id != request.tenant_id or version.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("version scope must match publish request")
    if version.work_item_id != request.work_item_id:
        raise CollaborativeActivitySourceMappingError("version work_item_id must match publish request")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    version_ref = _work_artifact_version_ref(published)
    execution = version.execution
    if isinstance(request, PublishWorkArtifactVersionFromExecutionRequest):
        execution = request.execution
    provenance = _execution_provenance_refs(execution)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_WORK,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.WORK_ARTIFACT_VERSION_PUBLISHED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=version.tenant_id,
            workspace_id=version.workspace_id,
            work_item_id=version.work_item_id,
        ),
        target=WorkArtifactVersionActivityTargetRef(version_ref=version_ref),
        outcome=_succeeded_outcome(),
        occurred_at=version.published_at,
        provenance_refs=provenance,
        requested_durability_class=requested_durability_class,
    )


def map_collaborative_decision_binding_created_publication(
    *,
    request: CreateCollaborativeDecisionBindingRequest,
    binding: CollaborativeDecisionBinding,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.AUDIT_CRITICAL
    ),
) -> CollaborativeActivityPublication:
    if binding.tenant_id != request.tenant_id or binding.workspace_id != request.workspace_id:
        raise CollaborativeActivitySourceMappingError("binding scope must match create request")
    if binding.work_item_id != request.work_item_id:
        raise CollaborativeActivitySourceMappingError("binding work_item_id must match create request")
    actor = _resolve_actor_from_request(request, principal_kind_resolver=principal_kind_resolver)
    decision_id = str(binding.decision_proposal.identity.decision_id)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=request.tenant_id,
            workspace_id=request.workspace_id,
            source=CollaborativeActivityBuiltinSource.COLLABORATIVE_DECISION_BINDING,
            source_stable_id=request.idempotency_key,
            activity_type=CollaborativeActivityBuiltinType.COLLABORATIVE_DECISION_BINDING_CREATED,
        ),
        actor=actor,
        scope=_scope_from_collaborative_work(
            tenant_id=binding.tenant_id,
            workspace_id=binding.workspace_id,
            work_item_id=binding.work_item_id,
        ),
        target=CollaborativeDecisionBindingActivityTargetRef(binding_id=binding.binding_id),
        outcome=_succeeded_outcome(),
        occurred_at=binding.created_at,
        provenance_refs=(DecisionActivityProvenanceRef(decision_id=decision_id),),
        requested_durability_class=requested_durability_class,
    )


def map_context_view_composed_publication(
    *,
    composition_request: ContextViewCompositionRequest,
    view: ContextView,
    principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    composition_completed_at: datetime,
    requested_durability_class: CollaborativeActivityDurabilityClass = (
        CollaborativeActivityDurabilityClass.INFORMATIONAL
    ),
) -> CollaborativeActivityPublication:
    request = composition_request.request
    scope = request.scope
    if view.view_id.strip() == "":
        raise CollaborativeActivitySourceMappingError("context view view_id is required")
    if view.scope.tenant_id != scope.tenant_id or view.scope.workspace_id != scope.workspace_id:
        raise CollaborativeActivitySourceMappingError("composed view scope must match composition request")
    if view.acting_principal_id != request.acting_principal_id:
        raise CollaborativeActivitySourceMappingError(
            "composed view acting_principal_id must match composition request",
        )
    if request.delegator_principal_id is not None and request.delegation is None:
        raise CollaborativeActivitySourceMappingError(
            "delegator_principal_id without canonical delegation object is insufficient for actor mapping",
        )
    principal_kind = principal_kind_resolver.resolve_principal_kind(
        tenant_id=scope.tenant_id,
        acting_principal_id=request.acting_principal_id,
    )
    actor = CollaborativeActivityActorRef(
        tenant_id=scope.tenant_id,
        principal_id=request.acting_principal_id,
        principal_kind=principal_kind,
        delegation_id=(
            request.delegation.delegation_id if request.delegation is not None else None
        ),
        delegator_principal_id=(
            request.delegation.delegator_principal_id if request.delegation is not None else None
        ),
    )
    work_item_id = scope.work_item_id
    activity_scope = CollaborativeActivityScope(
        tenant_id=scope.tenant_id,
        workspace_id=scope.workspace_id,
        work_item_id=work_item_id,
    )
    correlation = _correlation_from_operation_id(request.operation_id)
    return CollaborativeActivityPublication(
        idempotency_key=_activity_idempotency_key(
            tenant_id=scope.tenant_id,
            workspace_id=scope.workspace_id,
            source=CollaborativeActivityBuiltinSource.CONTEXT_VIEW,
            source_stable_id=view.view_id,
            activity_type=CollaborativeActivityBuiltinType.CONTEXT_VIEW_COMPOSED,
        ),
        actor=actor,
        scope=activity_scope,
        target=ContextViewActivityTargetRef(view_id=view.view_id),
        outcome=_succeeded_outcome(),
        occurred_at=composition_completed_at,
        provenance_refs=(ContextViewActivityProvenanceRef(view_id=view.view_id),),
        correlation=correlation,
        requested_durability_class=requested_durability_class,
    )


class CollaborativeWorkActivitySourceMapper(Protocol):
    """Replaceable mapping strategy for Collaborative Work authoritative mutations."""

    def map_work_item_created(
        self,
        *,
        request: CreateWorkItemRequest,
        work_item: WorkItem,
    ) -> CollaborativeActivityPublication: ...

    def map_work_item_state_changed(
        self,
        *,
        request: TransitionWorkItemRequest,
        work_item: WorkItem,
    ) -> CollaborativeActivityPublication: ...

    def map_assignment_created(
        self,
        *,
        request: CreateAssignmentRequest,
        assignment: Assignment,
    ) -> CollaborativeActivityPublication: ...

    def map_assignment_state_changed(
        self,
        *,
        request: TransitionAssignmentRequest,
        assignment: Assignment,
    ) -> CollaborativeActivityPublication: ...

    def map_work_artifact_created(
        self,
        *,
        request: CreateWorkArtifactRequest | CreateWorkArtifactFromExecutionRequest,
        published: PublishedWorkArtifactPublicationResult,
    ) -> CollaborativeActivityPublication: ...

    def map_work_artifact_version_published(
        self,
        *,
        request: PublishWorkArtifactVersionRequest | PublishWorkArtifactVersionFromExecutionRequest,
        published: PublishedWorkArtifactPublicationResult,
    ) -> CollaborativeActivityPublication: ...

    def map_decision_binding_created(
        self,
        *,
        request: CreateCollaborativeDecisionBindingRequest,
        binding: CollaborativeDecisionBinding,
    ) -> CollaborativeActivityPublication: ...


class ContextViewActivitySourceMapper(Protocol):
    def map_context_view_composed(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        view: ContextView,
        composition_completed_at: datetime,
    ) -> CollaborativeActivityPublication: ...


class FixedCollaborativeActivityActorPrincipalKindResolver:
    """Test / composition helper — returns one configured kind for all principals."""

    def __init__(self, *, principal_kind: PrincipalKind) -> None:
        self._principal_kind = principal_kind

    def resolve_principal_kind(
        self,
        *,
        tenant_id: str,
        acting_principal_id: str,
    ) -> PrincipalKind:
        _ = tenant_id, acting_principal_id
        return self._principal_kind


class DefaultCollaborativeWorkActivitySourceMapper:
    """Platform default Collaborative Work → ``CollaborativeActivityPublication`` mapping."""

    def __init__(
        self,
        *,
        principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    ) -> None:
        self._principal_kind_resolver = principal_kind_resolver

    def map_work_item_created(
        self,
        *,
        request: CreateWorkItemRequest,
        work_item: WorkItem,
    ) -> CollaborativeActivityPublication:
        return map_work_item_created_publication(
            request=request,
            work_item=work_item,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_work_item_state_changed(
        self,
        *,
        request: TransitionWorkItemRequest,
        work_item: WorkItem,
    ) -> CollaborativeActivityPublication:
        return map_work_item_state_changed_publication(
            request=request,
            work_item=work_item,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_assignment_created(
        self,
        *,
        request: CreateAssignmentRequest,
        assignment: Assignment,
    ) -> CollaborativeActivityPublication:
        return map_assignment_created_publication(
            request=request,
            assignment=assignment,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_assignment_state_changed(
        self,
        *,
        request: TransitionAssignmentRequest,
        assignment: Assignment,
    ) -> CollaborativeActivityPublication:
        return map_assignment_state_changed_publication(
            request=request,
            assignment=assignment,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_work_artifact_created(
        self,
        *,
        request: CreateWorkArtifactRequest | CreateWorkArtifactFromExecutionRequest,
        published: PublishedWorkArtifactPublicationResult,
    ) -> CollaborativeActivityPublication:
        return map_work_artifact_created_publication(
            request=request,
            published=published,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_work_artifact_version_published(
        self,
        *,
        request: PublishWorkArtifactVersionRequest | PublishWorkArtifactVersionFromExecutionRequest,
        published: PublishedWorkArtifactPublicationResult,
    ) -> CollaborativeActivityPublication:
        return map_work_artifact_version_published_publication(
            request=request,
            published=published,
            principal_kind_resolver=self._principal_kind_resolver,
        )

    def map_decision_binding_created(
        self,
        *,
        request: CreateCollaborativeDecisionBindingRequest,
        binding: CollaborativeDecisionBinding,
    ) -> CollaborativeActivityPublication:
        return map_collaborative_decision_binding_created_publication(
            request=request,
            binding=binding,
            principal_kind_resolver=self._principal_kind_resolver,
        )


class DefaultContextViewActivitySourceMapper:
    def __init__(
        self,
        *,
        principal_kind_resolver: CollaborativeActivityActorPrincipalKindResolver,
    ) -> None:
        self._principal_kind_resolver = principal_kind_resolver

    def map_context_view_composed(
        self,
        *,
        composition_request: ContextViewCompositionRequest,
        view: ContextView,
        composition_completed_at: datetime,
    ) -> CollaborativeActivityPublication:
        return map_context_view_composed_publication(
            composition_request=composition_request,
            view=view,
            principal_kind_resolver=self._principal_kind_resolver,
            composition_completed_at=composition_completed_at,
        )


__all__ = [
    "CollaborativeActivityActorPrincipalKindResolver",
    "CollaborativeActivitySourceMappingError",
    "CollaborativeWorkActivitySourceMapper",
    "ContextViewActivitySourceMapper",
    "DefaultCollaborativeWorkActivitySourceMapper",
    "DefaultContextViewActivitySourceMapper",
    "FixedCollaborativeActivityActorPrincipalKindResolver",
    "actor_ref_from_collaborative_work_request",
    "map_assignment_created_publication",
    "map_assignment_state_changed_publication",
    "map_collaborative_decision_binding_created_publication",
    "map_context_view_composed_publication",
    "map_work_artifact_created_publication",
    "map_work_artifact_version_published_publication",
    "map_work_item_created_publication",
    "map_work_item_state_changed_publication",
]
