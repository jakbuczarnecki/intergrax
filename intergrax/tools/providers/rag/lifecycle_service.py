# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.rag.lifecycle_contracts import (
    RagDeleteDocumentsInput,
    RagDeleteDocumentsOutput,
    RagDescribeCollectionInput,
    RagDescribeCollectionOutput,
)
from intergrax.rag.graph.lifecycle.graph_lifecycle_sync import sync_graph_delete_documents
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_DELETE_DOCUMENTS_TOOL_ID = "rag.delete_documents"
RAG_DESCRIBE_COLLECTION_TOOL_ID = "rag.describe_collection"


def perform_rag_delete_documents(
    ctx: ToolWiringContext,
    params: RagDeleteDocumentsInput,
) -> RagDeleteDocumentsOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagDeleteDocumentsOutput(used=False, reason="vectorstore_manager_not_configured")

    ids = [item.strip() for item in params.document_ids if item.strip()]
    if not ids:
        return RagDeleteDocumentsOutput(used=False, reason="document_ids_empty")

    try:
        vectorstore.delete(ids)
    except Exception as exc:
        return RagDeleteDocumentsOutput(
            used=False,
            reason=f"delete_error:{exc.__class__.__name__}",
        )

    graph_removed = sync_graph_delete_documents(ctx.rag_graph_store, ids)
    reason = "ok" if graph_removed == 0 else f"ok:graph_unlinked={graph_removed}"
    return RagDeleteDocumentsOutput(used=True, deleted_count=len(ids), reason=reason)


def perform_rag_describe_collection(
    ctx: ToolWiringContext,
    params: RagDescribeCollectionInput,
) -> RagDescribeCollectionOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagDescribeCollectionOutput(used=False, reason="vectorstore_manager_not_configured")

    collections: list[str] = []
    list_fn = getattr(vectorstore, "list_collections", None)
    if list_fn is not None:
        try:
            collections = [str(name) for name in list(list_fn())]
        except Exception as exc:
            return RagDescribeCollectionOutput(
                used=False,
                reason=f"list_collections_error:{exc.__class__.__name__}",
            )

    collection = params.collection.strip()
    if not collection and collections:
        collection = collections[0]

    count_fn = getattr(vectorstore, "count", None)
    if count_fn is None:
        return RagDescribeCollectionOutput(
            used=False,
            collections=collections,
            collection=collection,
            reason="count_not_supported",
        )

    try:
        document_count = int(count_fn())
    except Exception as exc:
        return RagDescribeCollectionOutput(
            used=False,
            collections=collections,
            collection=collection,
            reason=f"count_error:{exc.__class__.__name__}",
        )

    return RagDescribeCollectionOutput(
        used=True,
        collection=collection,
        document_count=document_count,
        collections=collections,
        reason="ok",
    )
