# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from intergrax.tools.providers.rag.list_collections_contracts import (
    RagListCollectionsInput,
    RagListCollectionsOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_LIST_COLLECTIONS_TOOL_ID = "rag.list_collections"


def perform_rag_list_collections(
    ctx: ToolWiringContext,
    _params: RagListCollectionsInput,
) -> RagListCollectionsOutput:
    vectorstore = ctx.vectorstore_manager
    if vectorstore is None:
        return RagListCollectionsOutput(used=False, reason="vectorstore_manager_not_configured")

    list_fn = attribute_access.optional(vectorstore, "list_collections", None)
    if list_fn is None:
        store = attribute_access.optional(vectorstore, "_store", None)
        list_fn = attribute_access.optional(store, "list_collections", None) if store is not None else None

    if list_fn is None:
        return RagListCollectionsOutput(used=False, reason="list_collections_not_supported")

    try:
        names = list(list_fn())
    except Exception as exc:
        return RagListCollectionsOutput(used=False, reason=f"list_collections_error:{exc.__class__.__name__}")

    return RagListCollectionsOutput(used=True, collections=[str(name) for name in names], reason="ok")
