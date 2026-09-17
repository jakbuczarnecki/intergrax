# © Artur Czarnecki. All rights reserved.

"""Default Knowledge reference reader — scoped retrieval projection without hydration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.knowledge.contracts.knowledge_reference_read import (
    KnowledgeChunkCanonicalRef,
    KnowledgeReferenceReadOutcome,
    KnowledgeReferenceReadRequest,
    KnowledgeReferenceReadResult,
    KnowledgeReferenceReadScope,
    validate_knowledge_reference_read_request,
)
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult
from intergrax.rag.retrieval.retrieval_errors import RetrievalError
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    MetadataMembershipCondition,
    VectorStoreScope,
)

__all__ = [
    "DefaultKnowledgeReferenceReader",
    "KnowledgeReferenceReadCapabilityBinding",
    "KnowledgeReferenceReadConfigurationError",
    "KnowledgeReferenceRetrievalBackend",
]


class KnowledgeReferenceReadConfigurationError(ValueError):
    """Default reader wiring violates mandatory capability authority invariants."""


@dataclass(frozen=True, slots=True)
class KnowledgeReferenceReadCapabilityBinding:
    """Authoritative tenant/workspace binding for a configured retrieval surface."""

    tenant_id: str
    workspace_id: str
    namespace: str | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        workspace = (self.workspace_id or "").strip()
        if not tenant:
            raise KnowledgeReferenceReadConfigurationError(
                "tenant_id must be non-empty"
            )
        if not workspace:
            raise KnowledgeReferenceReadConfigurationError(
                "workspace_id must be non-empty"
            )
        namespace = self.namespace
        if namespace is not None:
            namespace = namespace.strip()
            if not namespace:
                raise KnowledgeReferenceReadConfigurationError(
                    "namespace when set must be non-empty"
                )
        object.__setattr__(self, "tenant_id", tenant)
        object.__setattr__(self, "workspace_id", workspace)
        object.__setattr__(self, "namespace", namespace)


@runtime_checkable
class KnowledgeReferenceRetrievalBackend(Protocol):
    """Public retrieval surface used for reference projection (``RetrievalService``-compatible)."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


def _vector_store_scope(scope: KnowledgeReferenceReadScope) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id=scope.tenant_id,
        namespace=scope.namespace,
        workspace_id=scope.workspace_id,
    )


def _metadata_filter_for_resource(
    request: KnowledgeReferenceReadRequest,
) -> MetadataFilter | None:
    resource = request.scope.resource
    if resource is None:
        return None
    conditions: dict[str, str] = {}
    membership: tuple[MetadataMembershipCondition, ...] = ()
    if resource.document_id is not None:
        conditions["document_id"] = resource.document_id
    if resource.source_id is not None:
        membership = (
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=(resource.source_id,),
            ),
        )
    if not conditions and not membership:
        return None
    return MetadataFilter(conditions=conditions, membership=membership)


def _chunk_matches_scope(
    chunk: RetrievalChunk,
    *,
    scope: KnowledgeReferenceReadScope,
) -> bool:
    chunk_scope = chunk.scope or {}
    tenant = chunk_scope.get("tenant_id")
    workspace = chunk_scope.get("workspace_id")
    namespace = chunk_scope.get("namespace")
    if tenant != scope.tenant_id:
        return False
    if workspace != scope.workspace_id:
        return False
    if scope.namespace is not None and namespace != scope.namespace:
        return False
    resource = scope.resource
    if resource is None:
        return True
    provenance = chunk.provenance or {}
    if resource.source_id is not None and provenance.get("source_id") != resource.source_id:
        return False
    if resource.document_id is not None:
        root_id = provenance.get("root_document_id") or chunk.id
        if resource.document_id not in {chunk.id, root_id}:
            return False
    return True


def _chunk_to_canonical_ref(
    chunk: RetrievalChunk,
    *,
    tenant_id: str,
) -> KnowledgeChunkCanonicalRef:
    knowledge_ref = (chunk.vector_id or chunk.id or "").strip()
    provenance = chunk.provenance or {}
    source_id = provenance.get("source_id")
    source = source_id if isinstance(source_id, str) and source_id.strip() else None
    score = float(chunk.score) if chunk.score is not None else None
    return KnowledgeChunkCanonicalRef(
        tenant_id=tenant_id,
        knowledge_ref=knowledge_ref,
        document_id=chunk.id,
        source_id=source,
        rank=int(chunk.rank),
        relevance_score=score,
    )


def _capability_binding_rejects_request(
    binding: KnowledgeReferenceReadCapabilityBinding,
    identity: RequestIdentity,
    request: KnowledgeReferenceReadRequest,
) -> bool:
    if binding.tenant_id != identity.tenant_id:
        return True
    if binding.tenant_id != request.scope.tenant_id:
        return True
    if binding.workspace_id != request.scope.workspace_id:
        return True
    if binding.namespace is not None:
        if request.scope.namespace != binding.namespace:
            return True
    return False


@dataclass
class DefaultKnowledgeReferenceReader:
    """Project canonical references from the unified retrieval pipeline."""

    retrieval: KnowledgeReferenceRetrievalBackend | None = None
    capability_binding: KnowledgeReferenceReadCapabilityBinding | None = None

    def read_references(
        self,
        identity: RequestIdentity,
        request: KnowledgeReferenceReadRequest,
    ) -> KnowledgeReferenceReadResult:
        invalid = validate_knowledge_reference_read_request(identity, request)
        if invalid is not None:
            return KnowledgeReferenceReadResult(outcome=invalid, reason="identity_scope")

        if self.retrieval is None:
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.UNAVAILABLE,
                reason="retrieval_backend_not_configured",
            )

        binding = self.capability_binding
        if binding is None:
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.UNAVAILABLE,
                reason="capability_binding_not_configured",
            )

        if _capability_binding_rejects_request(binding, identity, request):
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.SCOPE_REJECTED,
                reason="capability_workspace_binding",
            )

        operation_scope = _vector_store_scope(request.scope)
        metadata_filter = _metadata_filter_for_resource(request)

        try:
            result = self.retrieval.retrieve(
                RetrievalRequest(
                    query=request.query.query_text,
                    final_top_k=request.query.limit,
                    scope=operation_scope,
                    metadata_filter=metadata_filter,
                )
            )
        except RetrievalError as exc:
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.UNAVAILABLE,
                reason=exc.kind.value,
            )
        except Exception:
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.UNAVAILABLE,
                reason="backend_error",
            )

        if not result.used and result.reason == "empty_query":
            return KnowledgeReferenceReadResult(
                outcome=KnowledgeReferenceReadOutcome.INVALID_REQUEST,
                reason="empty_query",
            )

        refs: list[KnowledgeChunkCanonicalRef] = []
        for chunk in result.chunks:
            if not _chunk_matches_scope(chunk, scope=request.scope):
                continue
            refs.append(
                _chunk_to_canonical_ref(chunk, tenant_id=request.scope.tenant_id)
            )
            if len(refs) >= request.query.limit:
                break

        return KnowledgeReferenceReadResult(
            outcome=KnowledgeReferenceReadOutcome.OK,
            references=tuple(refs),
            reason="ok" if refs or result.used else result.reason or "ok",
        )
