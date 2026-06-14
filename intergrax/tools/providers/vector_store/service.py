# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from intergrax.tools.providers.vector_store.contracts import (
    VectorStoreCountInput,
    VectorStoreCountOutput,
    VectorStoreDeleteInput,
    VectorStoreDeleteOutput,
    VectorStoreHealthInput,
    VectorStoreHealthOutput,
    VectorStoreListCollectionsInput,
    VectorStoreListCollectionsOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

VECTOR_STORE_COUNT_TOOL_ID = "vector_store.count"
VECTOR_STORE_DELETE_TOOL_ID = "vector_store.delete"
VECTOR_STORE_LIST_COLLECTIONS_TOOL_ID = "vector_store.list_collections"
VECTOR_STORE_HEALTH_TOOL_ID = "vector_store.health"


def _vectorstore(ctx: ToolWiringContext):
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return None
    return vectorstore


def vector_store_count(ctx: ToolWiringContext, _params: VectorStoreCountInput) -> VectorStoreCountOutput:
    vectorstore = _vectorstore(ctx)
    if vectorstore is None:
        return VectorStoreCountOutput(used=False, reason="vectorstore_manager_not_configured")
    count_fn = attribute_access.optional(vectorstore, "count", None)
    if count_fn is None:
        return VectorStoreCountOutput(used=False, reason="count_not_supported")
    try:
        document_count = int(count_fn())
    except Exception as exc:
        return VectorStoreCountOutput(used=False, reason=f"count_error:{exc.__class__.__name__}")
    return VectorStoreCountOutput(used=True, document_count=document_count, reason="ok")


def vector_store_delete(ctx: ToolWiringContext, params: VectorStoreDeleteInput) -> VectorStoreDeleteOutput:
    vectorstore = _vectorstore(ctx)
    if vectorstore is None:
        return VectorStoreDeleteOutput(used=False, reason="vectorstore_manager_not_configured")
    ids = [item.strip() for item in params.document_ids if item.strip()]
    if not ids:
        return VectorStoreDeleteOutput(used=False, reason="document_ids_empty")
    delete_fn = attribute_access.optional(vectorstore, "delete", None)
    if delete_fn is None:
        return VectorStoreDeleteOutput(used=False, reason="delete_not_supported")
    try:
        delete_fn(ids)
    except Exception as exc:
        return VectorStoreDeleteOutput(used=False, reason=f"delete_error:{exc.__class__.__name__}")
    return VectorStoreDeleteOutput(used=True, deleted_count=len(ids), reason="ok")


def vector_store_list_collections(
    ctx: ToolWiringContext,
    _params: VectorStoreListCollectionsInput,
) -> VectorStoreListCollectionsOutput:
    vectorstore = _vectorstore(ctx)
    if vectorstore is None:
        return VectorStoreListCollectionsOutput(used=False, reason="vectorstore_manager_not_configured")
    list_fn = attribute_access.optional(vectorstore, "list_collections", None)
    if list_fn is None:
        return VectorStoreListCollectionsOutput(used=False, reason="list_collections_not_supported")
    try:
        collections = [str(name) for name in list(list_fn())]
    except Exception as exc:
        return VectorStoreListCollectionsOutput(used=False, reason=f"list_collections_error:{exc.__class__.__name__}")
    return VectorStoreListCollectionsOutput(used=True, collections=collections, reason="ok")


def vector_store_health(ctx: ToolWiringContext, _params: VectorStoreHealthInput) -> VectorStoreHealthOutput:
    count_out = vector_store_count(ctx, VectorStoreCountInput())
    if not count_out.used:
        return VectorStoreHealthOutput(used=True, healthy=False, reason=count_out.reason)
    return VectorStoreHealthOutput(
        used=True,
        healthy=True,
        document_count=count_out.document_count,
        reason="ok",
    )
