# © Artur Czarnecki. All rights reserved.

"""Resolve DocumentStore for LKW managed-workspace application state."""

from __future__ import annotations

import os

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore

_DEFAULT_COLLECTION = "lkw_managed_workspaces"


def resolve_managed_workspace_document_store(
    document_store: DocumentStore | None = None,
) -> DocumentStore:
    """Prefer injected store, then MongoDB when configured, else in-memory."""
    if document_store is not None:
        return document_store

    mongo_uri = (os.environ.get("INTERGRAX_MONGODB_URI") or "").strip()
    if mongo_uri:
        from intergrax.integrations.providers.document_store.mongodb.bundle import (
            create_mongodb_document_store,
        )

        collection_name = (
            os.environ.get("LKW_MANAGED_WORKSPACE_COLLECTION", _DEFAULT_COLLECTION).strip()
            or _DEFAULT_COLLECTION
        )
        return create_mongodb_document_store(collection_name=collection_name)

    return InMemoryDocumentStore()
