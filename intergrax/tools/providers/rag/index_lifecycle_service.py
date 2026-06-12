# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.rag.index_lifecycle_contracts import (
    RagCheckIndexStatusInput,
    RagCheckIndexStatusOutput,
    RagDocumentSummaryOutput,
    RagGetDocumentInput,
    RagGetDocumentOutput,
    RagListDocumentsInput,
    RagListDocumentsOutput,
    RagMetadataMatchOutput,
    RagPurgeCollectionInput,
    RagPurgeCollectionOutput,
    RagSearchByMetadataInput,
    RagSearchByMetadataOutput,
)
from intergrax.rag.graph.lifecycle.graph_lifecycle_sync import sync_graph_purge_collection
from intergrax.tools.registry.runtime_bindings import VectorstoreIndexLifecycleBinding
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_LIST_DOCUMENTS_TOOL_ID = "rag.list_documents"
RAG_GET_DOCUMENT_TOOL_ID = "rag.get_document"
RAG_CHECK_INDEX_STATUS_TOOL_ID = "rag.check_index_status"
RAG_SEARCH_BY_METADATA_TOOL_ID = "rag.search_by_metadata"
RAG_PURGE_COLLECTION_TOOL_ID = "rag.purge_collection"


def perform_rag_list_documents(
    ctx: ToolWiringContext,
    params: RagListDocumentsInput,
) -> RagListDocumentsOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagListDocumentsOutput(used=False, reason="vectorstore_manager_not_configured")
    if not isinstance(vectorstore, VectorstoreIndexLifecycleBinding):
        return RagListDocumentsOutput(used=False, reason="list_documents_not_supported")

    try:
        document_ids = [
            str(item)
            for item in vectorstore.list_document_ids(limit=params.limit, offset=params.offset)
        ]
    except RuntimeError as exc:
        if "not_supported" in str(exc):
            return RagListDocumentsOutput(used=False, reason="list_documents_not_supported")
        return RagListDocumentsOutput(
            used=False,
            reason=f"list_documents_error:{exc.__class__.__name__}",
        )
    except Exception as exc:
        return RagListDocumentsOutput(
            used=False,
            reason=f"list_documents_error:{exc.__class__.__name__}",
        )

    documents = [RagDocumentSummaryOutput(document_id=item) for item in document_ids]
    return RagListDocumentsOutput(
        used=True,
        documents=documents,
        total=len(documents),
        reason="ok",
    )


def perform_rag_get_document(
    ctx: ToolWiringContext,
    params: RagGetDocumentInput,
) -> RagGetDocumentOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagGetDocumentOutput(used=False, reason="vectorstore_manager_not_configured")
    if not isinstance(vectorstore, VectorstoreIndexLifecycleBinding):
        return RagGetDocumentOutput(used=False, reason="get_document_not_supported")

    try:
        payload = vectorstore.get_document(params.document_id.strip())
    except RuntimeError as exc:
        if "not_supported" in str(exc):
            return RagGetDocumentOutput(used=False, reason="get_document_not_supported")
        return RagGetDocumentOutput(
            used=False,
            document_id=params.document_id.strip(),
            reason=f"get_document_error:{exc.__class__.__name__}",
        )
    except Exception as exc:
        return RagGetDocumentOutput(
            used=False,
            document_id=params.document_id.strip(),
            reason=f"get_document_error:{exc.__class__.__name__}",
        )

    if payload is None:
        return RagGetDocumentOutput(
            used=False,
            document_id=params.document_id.strip(),
            reason="document_not_found",
        )

    return RagGetDocumentOutput(
        used=True,
        document_id=str(payload.get("id") or params.document_id.strip()),
        text=str(payload.get("text") or ""),
        metadata=dict(payload.get("metadata") or {}),
        reason="ok",
    )


def perform_rag_check_index_status(
    ctx: ToolWiringContext,
    params: RagCheckIndexStatusInput,
) -> RagCheckIndexStatusOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagCheckIndexStatusOutput(used=False, reason="vectorstore_manager_not_configured")
    if not isinstance(vectorstore, VectorstoreIndexLifecycleBinding):
        return RagCheckIndexStatusOutput(used=False, reason="count_not_supported")

    try:
        collections = [str(name) for name in vectorstore.list_collections()]
    except Exception as exc:
        return RagCheckIndexStatusOutput(
            used=False,
            reason=f"list_collections_error:{exc.__class__.__name__}",
        )

    collection = params.collection.strip()
    if not collection and collections:
        collection = collections[0]

    try:
        document_count = int(vectorstore.count())
    except Exception as exc:
        return RagCheckIndexStatusOutput(
            used=False,
            collections=collections,
            collection=collection,
            reason=f"count_error:{exc.__class__.__name__}",
        )

    return RagCheckIndexStatusOutput(
        used=True,
        ready=document_count > 0,
        collection=collection,
        document_count=document_count,
        collections=collections,
        reason="ok",
    )


def perform_rag_search_by_metadata(
    ctx: ToolWiringContext,
    params: RagSearchByMetadataInput,
) -> RagSearchByMetadataOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagSearchByMetadataOutput(used=False, reason="vectorstore_manager_not_configured")
    if not isinstance(vectorstore, VectorstoreIndexLifecycleBinding):
        return RagSearchByMetadataOutput(used=False, reason="search_by_metadata_not_supported")

    conditions = dict(params.filters)
    tenant_id = params.tenant_id.strip()
    if tenant_id:
        conditions["tenant_id"] = tenant_id

    try:
        raw_matches = vectorstore.search_by_metadata(conditions=conditions, limit=params.limit)
    except ValueError as exc:
        return RagSearchByMetadataOutput(
            used=False,
            reason=f"search_by_metadata_error:{exc.__class__.__name__}",
        )
    except Exception as exc:
        return RagSearchByMetadataOutput(
            used=False,
            reason=f"search_by_metadata_error:{exc.__class__.__name__}",
        )

    matches = [
        RagMetadataMatchOutput(
            document_id=str(item.get("id") or ""),
            text=str(item.get("text") or ""),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw_matches
    ]
    return RagSearchByMetadataOutput(
        used=True,
        matches=matches,
        total=len(matches),
        reason="ok",
    )


def perform_rag_purge_collection(
    ctx: ToolWiringContext,
    params: RagPurgeCollectionInput,
) -> RagPurgeCollectionOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagPurgeCollectionOutput(used=False, reason="vectorstore_manager_not_configured")
    if not isinstance(vectorstore, VectorstoreIndexLifecycleBinding):
        return RagPurgeCollectionOutput(used=False, reason="purge_collection_not_supported")

    try:
        result = vectorstore.purge_collection(
            dry_run=params.dry_run,
            tenant_id=params.tenant_id.strip(),
        )
    except ValueError as exc:
        return RagPurgeCollectionOutput(
            used=False,
            reason=f"purge_collection_error:{exc.__class__.__name__}",
        )
    except Exception as exc:
        return RagPurgeCollectionOutput(
            used=False,
            reason=f"purge_collection_error:{exc.__class__.__name__}",
        )

    would_delete = int(result.get("would_delete") or 0)
    deleted = int(result.get("deleted") or 0)
    graph_purged = 0
    if not params.dry_run:
        graph_purged = sync_graph_purge_collection(
            ctx.rag_graph_store,
            tenant_id=params.tenant_id.strip() or None,
        )
    reason = "ok" if graph_purged == 0 else f"ok:graph_purged={graph_purged}"
    return RagPurgeCollectionOutput(
        used=True,
        dry_run=bool(result.get("dry_run", params.dry_run)),
        would_delete=would_delete,
        deleted=deleted,
        tenant_id=str(result.get("tenant_id") or params.tenant_id.strip()),
        reason=reason,
    )
