# © Artur Czarnecki. All rights reserved.

"""Pure contract-to-contract mapping helpers for MP-5F-B5 source adapters."""

from __future__ import annotations

from enum import Enum

from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkArtifactCanonicalRef,
    CollaborativeWorkItemCanonicalRef,
    CollaborativeWorkReferenceEntityKind,
    CollaborativeWorkReferenceReadQuery,
    CollaborativeWorkReferenceReadRequest,
    CollaborativeWorkReferenceReadScope,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.context_view import (
    ContextViewCollaborativeWorkSourceRef,
    ContextViewKnowledgeSourceRef,
    ContextViewMemorySourceRef,
    ContextViewOperationScope,
    ContextViewScope,
    ContextViewUclSourceRef,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_source_ports import (
    ContextViewCollaborativeWorkSourceRequest,
    ContextViewKnowledgeSourceRequest,
    ContextViewMemorySourceRequest,
    ContextViewSourceOutcome,
    ContextViewSourceRequestIdentityView,
    ContextViewUclSourceRequest,
)
from intergrax.contracts.context_view_visibility_policy import (
    suggested_context_view_source_visibility,
)
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadQuery,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadScope,
    KnowledgeScopedResourceRef,
)
from intergrax.memory.contracts.memory_reference_read import (
    MemoryRecordCanonicalRef,
    MemoryReferenceReadQuery,
    MemoryReferenceReadRequest,
    MemoryReferenceReadScope,
    MemoryScopedResourceRef,
)
from intergrax.ucl.contracts.ucl_reference_read import (
    UclOptimizationArtifactCanonicalRef,
    UclReferenceReadQuery,
    UclReferenceReadRequest,
    UclReferenceReadScope,
    UclScopedResourceRef,
    format_ucl_artifact_locator,
)

MEMORY_RECORD_REF_PREFIX: str = "memory-record/v1/"


class ContextViewSourceAdapterConfigurationError(ValueError):
    """Adapter or composition wiring violates mandatory DI invariants."""


def suggested_visibility_from_request(
    request: ContextViewSourceRequestIdentityView,
) -> ContextViewVisibilityClass:
    return suggested_context_view_source_visibility(request.eligible_visibility_classes)


def request_identity_from_source_request(
    request: ContextViewSourceRequestIdentityView,
) -> RequestIdentity:
    return request.principal_identity


def map_domain_read_outcome_to_context_view(outcome: Enum) -> ContextViewSourceOutcome:
    name = outcome.name if isinstance(outcome, Enum) else str(outcome)
    if name == "OK":
        return ContextViewSourceOutcome.OK
    if name == "UNAVAILABLE":
        return ContextViewSourceOutcome.SOURCE_UNAVAILABLE
    if name in {"SCOPE_REJECTED", "ACCESS_DENIED"}:
        return ContextViewSourceOutcome.SCOPE_REJECTED
    if name == "INVALID_REQUEST":
        return ContextViewSourceOutcome.INVALID_REQUEST
    return ContextViewSourceOutcome.SOURCE_UNAVAILABLE


def format_context_view_memory_record_ref(ref: MemoryRecordCanonicalRef) -> str:
    return f"{MEMORY_RECORD_REF_PREFIX}{ref.memory_id}@{ref.revision}"


def parse_context_view_memory_record_ref(
    *,
    tenant_id: str,
    record_ref: str,
) -> MemoryRecordCanonicalRef | None:
    if not record_ref.startswith(MEMORY_RECORD_REF_PREFIX):
        return None
    body = record_ref[len(MEMORY_RECORD_REF_PREFIX) :]
    if "@" not in body:
        return None
    memory_id, revision_text = body.rsplit("@", 1)
    memory_id = memory_id.strip()
    if not memory_id:
        return None
    try:
        revision = int(revision_text)
    except ValueError:
        return None
    try:
        return MemoryRecordCanonicalRef(
            tenant_id=tenant_id,
            memory_id=memory_id,
            revision=revision,
        )
    except Exception:
        return None


def map_memory_record_to_context_view_ref(
    ref: MemoryRecordCanonicalRef,
) -> ContextViewMemorySourceRef:
    return ContextViewMemorySourceRef(
        tenant_id=ref.tenant_id,
        record_ref=format_context_view_memory_record_ref(ref),
    )


def _memory_scope_user_id(identity: RequestIdentity) -> str | None:
    user_id = (identity.user_id or "").strip()
    if user_id:
        return user_id
    return None


def memory_read_request_from_context_view(
    request: ContextViewMemorySourceRequest,
    *,
    identity: RequestIdentity,
) -> MemoryReferenceReadRequest:
    scope = request.scope
    return MemoryReferenceReadRequest(
        scope=MemoryReferenceReadScope(
            tenant_id=scope.tenant_id,
            workspace_id=scope.workspace_id,
            user_id=_memory_scope_user_id(identity),
            resource=None,
        ),
        query=MemoryReferenceReadQuery(),
    )


def candidate_scope_from_memory_evaluated(
    evaluated_scope: MemoryReferenceReadScope,
) -> ContextViewScope:
    work_item_id: str | None = None
    resource = evaluated_scope.resource
    if resource is not None and resource.resource_kind == "work_item":
        work_item_id = resource.resource_id
    return ContextViewScope(
        tenant_id=evaluated_scope.tenant_id,
        workspace_id=evaluated_scope.workspace_id,
        work_item_id=work_item_id,
        operation_scope=None,
    )


def memory_evaluated_scope_within_request(
    *,
    request: ContextViewMemorySourceRequest,
    evaluated_scope: MemoryReferenceReadScope,
) -> bool:
    scope = request.scope
    if evaluated_scope.tenant_id != scope.tenant_id:
        return False
    if evaluated_scope.workspace_id != scope.workspace_id:
        return False
    if scope.work_item_id is not None:
        resource = evaluated_scope.resource
        if resource is not None:
            if resource.resource_kind != "work_item":
                return False
            if resource.resource_id != scope.work_item_id:
                return False
    return True


def memory_ref_within_evaluated_scope(
    *,
    ref: MemoryRecordCanonicalRef,
    evaluated_scope: MemoryReferenceReadScope,
) -> bool:
    return ref.tenant_id == evaluated_scope.tenant_id


def knowledge_read_request_from_context_view(
    request: ContextViewKnowledgeSourceRequest,
) -> KnowledgeReferenceReadRequest:
    scope = request.scope
    resource: KnowledgeScopedResourceRef | None = None
    if scope.operation_scope is not None and scope.operation_scope.resource_scope is not None:
        resource = KnowledgeScopedResourceRef(
            document_id=scope.operation_scope.resource_scope,
        )
    return KnowledgeReferenceReadRequest(
        scope=KnowledgeReferenceReadScope(
            tenant_id=scope.tenant_id,
            workspace_id=scope.workspace_id,
            resource=resource,
        ),
        query=KnowledgeReferenceReadQuery(query_text=request.reference_read_query_text),
    )


def map_knowledge_chunk_to_context_view_ref(
    ref: KnowledgeChunkCanonicalRef,
) -> ContextViewKnowledgeSourceRef:
    return ContextViewKnowledgeSourceRef(
        tenant_id=ref.tenant_id,
        knowledge_ref=ref.knowledge_ref,
    )


def candidate_scope_from_knowledge_evaluated(
    evaluated_scope: KnowledgeReferenceReadScope,
    *,
    request_scope: ContextViewScope,
) -> ContextViewScope:
    operation_scope: ContextViewOperationScope | None = None
    resource = evaluated_scope.resource
    if resource is not None and resource.document_id is not None:
        request_operation = request_scope.operation_scope
        operation_id = (
            request_operation.operation_id
            if request_operation is not None
            else "knowledge.document"
        )
        operation_scope = ContextViewOperationScope(
            operation_id=operation_id,
            resource_scope=resource.document_id,
        )
    return ContextViewScope(
        tenant_id=evaluated_scope.tenant_id,
        workspace_id=evaluated_scope.workspace_id,
        operation_scope=operation_scope,
    )


def knowledge_evaluated_scope_within_request(
    *,
    request: ContextViewKnowledgeSourceRequest,
    evaluated_scope: KnowledgeReferenceReadScope,
) -> bool:
    scope = request.scope
    if evaluated_scope.tenant_id != scope.tenant_id:
        return False
    if evaluated_scope.workspace_id != scope.workspace_id:
        return False
    request_resource = scope.operation_scope.resource_scope if scope.operation_scope else None
    if request_resource is not None:
        evaluated_resource = evaluated_scope.resource
        if evaluated_resource is None or evaluated_resource.document_id != request_resource:
            return False
    return True


def knowledge_ref_within_evaluated_scope(
    *,
    ref: KnowledgeChunkCanonicalRef,
    evaluated_scope: KnowledgeReferenceReadScope,
) -> bool:
    if ref.tenant_id != evaluated_scope.tenant_id:
        return False
    resource = evaluated_scope.resource
    if resource is None:
        return True
    if resource.document_id is not None and ref.document_id != resource.document_id:
        return False
    if resource.source_id is not None and ref.source_id != resource.source_id:
        return False
    return True


def ucl_read_scope_from_context_view(
    scope: ContextViewScope,
) -> UclReferenceReadScope | None:
    operation = scope.operation_scope
    if operation is None or operation.resource_scope is None:
        return None
    return UclReferenceReadScope(
        tenant_id=scope.tenant_id,
        workspace_id=scope.workspace_id,
        context_scope_id=operation.resource_scope,
        resource=None,
    )


def ucl_read_request_from_context_view(
    request: ContextViewUclSourceRequest,
) -> UclReferenceReadRequest | None:
    read_scope = ucl_read_scope_from_context_view(request.scope)
    if read_scope is None:
        return None
    return UclReferenceReadRequest(scope=read_scope, query=UclReferenceReadQuery())


def map_ucl_ref_to_context_view_ref(
    ref: UclOptimizationArtifactCanonicalRef,
) -> ContextViewUclSourceRef:
    return ContextViewUclSourceRef(
        tenant_id=ref.tenant_id,
        ucl_artifact_ref=format_ucl_artifact_locator(ref),
    )


def ucl_ref_within_request_scope(
    *,
    ref: UclOptimizationArtifactCanonicalRef,
    request: ContextViewUclSourceRequest,
) -> bool:
    scope = request.scope
    if ref.tenant_id != scope.tenant_id:
        return False
    if ref.workspace_id != scope.workspace_id:
        return False
    operation = scope.operation_scope
    if operation is None or operation.resource_scope is None:
        return False
    if ref.context_scope_id != operation.resource_scope:
        return False
    return True


def candidate_scope_for_ucl_ref(
    *,
    ref: UclOptimizationArtifactCanonicalRef,
    request_scope: ContextViewScope,
) -> ContextViewScope:
    operation = request_scope.operation_scope
    narrowed_operation: ContextViewOperationScope | None = None
    if operation is not None:
        narrowed_operation = ContextViewOperationScope(
            operation_id=operation.operation_id,
            resource_scope=ref.context_scope_id,
        )
    return ContextViewScope(
        tenant_id=ref.tenant_id,
        workspace_id=ref.workspace_id,
        operation_scope=narrowed_operation,
    )


def collaborative_work_read_request_from_context_view(
    request: ContextViewCollaborativeWorkSourceRequest,
) -> CollaborativeWorkReferenceReadRequest:
    scope = request.scope
    return CollaborativeWorkReferenceReadRequest(
        scope=CollaborativeWorkReferenceReadScope(
            tenant_id=scope.tenant_id,
            workspace_id=scope.workspace_id,
            work_item_id=scope.work_item_id,
        ),
        query=CollaborativeWorkReferenceReadQuery(
            entity_kinds=frozenset(
                {
                    CollaborativeWorkReferenceEntityKind.WORK_ITEM,
                    CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT,
                    CollaborativeWorkReferenceEntityKind.WORK_ARTIFACT_VERSION,
                }
            ),
        ),
    )


def map_collaborative_work_item_ref(
    ref: CollaborativeWorkItemCanonicalRef,
) -> ContextViewCollaborativeWorkSourceRef:
    return ContextViewCollaborativeWorkSourceRef(
        tenant_id=ref.tenant_id,
        workspace_id=ref.workspace_id,
        work_item_id=ref.work_item_id,
    )


def map_collaborative_work_artifact_ref(
    ref: CollaborativeWorkArtifactCanonicalRef,
) -> ContextViewCollaborativeWorkSourceRef:
    return ContextViewCollaborativeWorkSourceRef(
        tenant_id=ref.tenant_id,
        workspace_id=ref.workspace_id,
        work_artifact_version=WorkArtifactVersionRef(
            tenant_id=ref.tenant_id,
            workspace_id=ref.workspace_id,
            work_item_id=ref.work_item_id,
            work_artifact_id=ref.work_artifact_id,
            work_artifact_version_id=ref.current_version_id,
        ),
    )


def map_collaborative_work_version_ref(
    ref: WorkArtifactVersionRef,
) -> ContextViewCollaborativeWorkSourceRef:
    return ContextViewCollaborativeWorkSourceRef(
        tenant_id=ref.tenant_id,
        workspace_id=ref.workspace_id,
        work_artifact_version=ref,
    )


def _collaborative_work_item_id(
    ref: (
        CollaborativeWorkItemCanonicalRef
        | CollaborativeWorkArtifactCanonicalRef
        | WorkArtifactVersionRef
    ),
) -> str:
    if isinstance(ref, CollaborativeWorkItemCanonicalRef):
        return ref.work_item_id
    return ref.work_item_id


def collaborative_work_ref_within_request_scope(
    *,
    ref: (
        CollaborativeWorkItemCanonicalRef
        | CollaborativeWorkArtifactCanonicalRef
        | WorkArtifactVersionRef
    ),
    request: ContextViewCollaborativeWorkSourceRequest,
) -> bool:
    scope = request.scope
    if ref.tenant_id != scope.tenant_id:
        return False
    if ref.workspace_id != scope.workspace_id:
        return False
    if scope.work_item_id is not None:
        if _collaborative_work_item_id(ref) != scope.work_item_id:
            return False
    return True


def candidate_scope_for_collaborative_work_ref(
    *,
    ref: (
        CollaborativeWorkItemCanonicalRef
        | CollaborativeWorkArtifactCanonicalRef
        | WorkArtifactVersionRef
    ),
    request_scope: ContextViewScope,
) -> ContextViewScope:
    work_item_id = _collaborative_work_item_id(ref)
    return ContextViewScope(
        tenant_id=ref.tenant_id,
        workspace_id=ref.workspace_id,
        work_item_id=work_item_id,
        operation_scope=request_scope.operation_scope,
    )
