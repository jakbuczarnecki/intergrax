# © Artur Czarnecki. All rights reserved.

"""Embedding and vector index identity capture for portability proof."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.tools.providers.rag.scope import resolve_tenant_scoped_vectorstore
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.utils import attribute_access

from local_workspace_application.model_runtime_proof.contracts import (
    EmbeddingIdentityRecord,
    IndexIdentityRecord,
)
from local_workspace_application.workspaces.models import WorkspaceOperationStatus
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


def resolve_collection_identity(
    wiring_context: ToolWiringContext,
    tenant_id: str,
) -> str | None:
    """Resolve physical/logical collection identity from the configured vector store."""
    manager = resolve_tenant_scoped_vectorstore(wiring_context, tenant_id)
    if manager is None:
        return None

    collection_name = attribute_access.optional(manager, "_collection_name", None)
    if collection_name is not None and str(collection_name).strip():
        return str(collection_name).strip()

    list_collections = getattr(manager, "list_collections", None)
    if callable(list_collections):
        try:
            raw_names = list_collections()
            names = [str(name) for name in cast(Any, raw_names)]
        except Exception:
            names = []
        if names:
            return ",".join(sorted(names))

    store = attribute_access.optional(manager, "_store", manager)
    if store is not None:
        store_collections = getattr(store, "list_collections", None)
        if callable(store_collections):
            try:
                raw_names = store_collections()
                names = [str(name) for name in cast(Any, raw_names)]
            except Exception:
                names = []
            if names:
                return ",".join(sorted(names))

        direct_name = attribute_access.optional(store, "collection_name", None)
        if direct_name is not None and str(direct_name).strip():
            return str(direct_name).strip()

        for config_attr in (
            "cfg",
            "store_config",
            "_store_config",
            "_config",
            "config",
        ):
            cfg = attribute_access.optional(store, config_attr, None)
            cfg_name = attribute_access.optional(cfg, "collection_name", None)
            if cfg_name is not None and str(cfg_name).strip():
                return str(cfg_name).strip()

    return None


def _resolve_chunk_count(
    repository: ManagedWorkspaceRepository,
    *,
    tenant_id: str,
    workspace_id: str,
    vector_count: int,
) -> int | None:
    operations = repository.list_ingestion_operations(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        statuses={WorkspaceOperationStatus.COMPLETED},
    )
    for operation in operations:
        indexed = getattr(operation, "documents_indexed", None)
        if indexed is not None:
            return int(indexed)
    return vector_count if vector_count > 0 else None


def index_identity_is_complete(identity: IndexIdentityRecord) -> bool:
    if not identity.collection_identity or identity.collection_identity == "unknown":
        return False
    if identity.source_id is None:
        return False
    if identity.document_id is None:
        return False
    if identity.content_hash is None:
        return False
    if identity.chunk_count is None:
        return False
    if identity.embedding.provider == "unknown":
        return False
    if identity.embedding.model == "unknown":
        return False
    return True


def capture_index_identity(
    *,
    tenant_id: str,
    workspace_id: str,
    repository: ManagedWorkspaceRepository,
    wiring_context: ToolWiringContext,
    embedding_manager: BaseEmbeddingManager | None,
) -> IndexIdentityRecord:
    refs = repository.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
    source_id = refs[0].source_id if refs else None
    document_id = refs[0].document_id if refs else None
    content_hash = refs[0].content_hash if refs else None
    scoped = resolve_tenant_scoped_vectorstore(wiring_context, tenant_id)
    vector_count = (
        int(scoped.count()) if scoped is not None and hasattr(scoped, "count") else 0
    )
    collection = resolve_collection_identity(wiring_context, tenant_id)
    if collection is None:
        collection = "unknown"
    chunk_count = _resolve_chunk_count(
        repository,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        vector_count=vector_count,
    )
    return IndexIdentityRecord(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        source_id=source_id,
        document_id=document_id,
        content_hash=content_hash,
        collection_identity=collection,
        vector_count=vector_count,
        chunk_count=chunk_count,
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


@dataclass(frozen=True, slots=True)
class IndexIdentityComparison:
    collection_identity: bool
    vector_count: bool
    source_id: bool
    document_id: bool
    content_hash: bool
    chunk_count: bool


def compare_index_identity(
    before: IndexIdentityRecord,
    after: IndexIdentityRecord,
) -> IndexIdentityComparison:
    return IndexIdentityComparison(
        collection_identity=before.collection_identity == after.collection_identity,
        vector_count=before.vector_count == after.vector_count,
        source_id=before.source_id == after.source_id,
        document_id=before.document_id == after.document_id,
        content_hash=before.content_hash == after.content_hash,
        chunk_count=before.chunk_count == after.chunk_count,
    )
