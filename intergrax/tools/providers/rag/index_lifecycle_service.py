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
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_LIST_DOCUMENTS_TOOL_ID = "rag.list_documents"
RAG_GET_DOCUMENT_TOOL_ID = "rag.get_document"
RAG_CHECK_INDEX_STATUS_TOOL_ID = "rag.check_index_status"


def perform_rag_list_documents(
    ctx: ToolWiringContext,
    params: RagListDocumentsInput,
) -> RagListDocumentsOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagListDocumentsOutput(used=False, reason="vectorstore_manager_not_configured")

    list_fn = getattr(vectorstore, "list_document_ids", None)
    if list_fn is None:
        return RagListDocumentsOutput(used=False, reason="list_documents_not_supported")

    try:
        document_ids = [str(item) for item in list(list_fn(limit=params.limit, offset=params.offset))]
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

    get_fn = getattr(vectorstore, "get_document", None)
    if get_fn is None:
        return RagGetDocumentOutput(used=False, reason="get_document_not_supported")

    try:
        payload = get_fn(params.document_id.strip())
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

    collections: list[str] = []
    list_fn = getattr(vectorstore, "list_collections", None)
    if list_fn is not None:
        try:
            collections = [str(name) for name in list(list_fn())]
        except Exception as exc:
            return RagCheckIndexStatusOutput(
                used=False,
                reason=f"list_collections_error:{exc.__class__.__name__}",
            )

    collection = params.collection.strip()
    if not collection and collections:
        collection = collections[0]

    count_fn = getattr(vectorstore, "count", None)
    if count_fn is None:
        return RagCheckIndexStatusOutput(
            used=False,
            collections=collections,
            collection=collection,
            reason="count_not_supported",
        )

    try:
        document_count = int(count_fn())
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
