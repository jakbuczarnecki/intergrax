# © Artur Czarnecki. All rights reserved.

"""Embedding and vector index identity capture for portability proof."""

from __future__ import annotations

from typing import Any

from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.tools.providers.rag.scope import resolve_tenant_scoped_vectorstore
from intergrax.tools.registry.wiring import ToolWiringContext

from local_workspace_application.model_runtime_proof.contracts import (
    EmbeddingIdentityRecord,
    IndexIdentityRecord,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository


def _provider_model_from_pipeline(pipeline: Any) -> tuple[str, str, int | None]:
    provider_id = str(getattr(pipeline, "_provider_id", "unknown"))
    engine = getattr(pipeline, "_engine", None)
    registry = getattr(engine, "_registry", None)
    model = "unknown"
    dimensions: int | None = None
    if registry is not None:
        try:
            provider = registry.get(provider_id)
            model = str(
                getattr(provider, "_model_name", None)
                or getattr(provider, "DEFAULT_MODEL", "unknown")
            )
            if hasattr(provider, "dimension"):
                dimensions = int(provider.dimension())
        except Exception:
            pass
    return provider_id, model, dimensions


def resolve_embedding_identity(
    embedding_manager: BaseEmbeddingManager | None,
) -> EmbeddingIdentityRecord:
    if embedding_manager is None:
        return EmbeddingIdentityRecord(
            provider="unknown", model="unknown", dimensions=None
        )
    pipeline = getattr(embedding_manager, "_pipeline", None)
    if pipeline is None:
        return EmbeddingIdentityRecord(
            provider="unknown", model="unknown", dimensions=None
        )
    provider_id, model, dimensions = _provider_model_from_pipeline(pipeline)
    return EmbeddingIdentityRecord(
        provider=provider_id, model=model, dimensions=dimensions
    )


def capture_index_identity(
    *,
    tenant_id: str,
    workspace_id: str,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext,
    embedding_manager: BaseEmbeddingManager | None,
    collection_identity: str | None = None,
) -> IndexIdentityRecord:
    refs = repository.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
    source_id = refs[0].source_id if refs else None
    document_id = refs[0].document_id if refs else None
    content_hash = refs[0].content_hash if refs else None
    scoped = resolve_tenant_scoped_vectorstore(wiring_context, tenant_id)
    vector_count = (
        int(scoped.count()) if scoped is not None and hasattr(scoped, "count") else 0
    )
    collection = collection_identity or workspace_id
    return IndexIdentityRecord(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        document_id=document_id,
        content_hash=content_hash,
        collection_identity=collection,
        vector_count=vector_count,
        chunk_count=vector_count if vector_count else None,
        embedding=resolve_embedding_identity(embedding_manager),
    )


def compare_embedding_identity(
    before: EmbeddingIdentityRecord,
    after: EmbeddingIdentityRecord,
) -> bool:
    return (
        before.provider == after.provider
        and before.model == after.model
        and before.dimensions == after.dimensions
    )


def compare_index_identity(
    before: IndexIdentityRecord,
    after: IndexIdentityRecord,
) -> tuple[bool, bool, bool]:
    collection_ok = before.collection_identity == after.collection_identity
    vector_ok = before.vector_count == after.vector_count
    document_ok = before.document_id == after.document_id
    return collection_ok, vector_ok, document_ok
