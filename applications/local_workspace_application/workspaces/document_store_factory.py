# © Artur Czarnecki. All rights reserved.

"""Resolve DocumentStore for LKW managed-workspace application state."""

from __future__ import annotations

import os
from typing import Literal

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore

_DEFAULT_COLLECTION = "lkw_managed_workspaces"


def resolve_managed_workspace_document_store(
    document_store: DocumentStore | None = None,
    *,
    backend: Literal["auto", "mongodb", "inmemory"] = "auto",
) -> DocumentStore:
    """Resolve the configured store without silently downgrading production."""
    if document_store is not None:
        return document_store

    configured_backend = (
        os.environ.get("LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND", backend)
        or "auto"
    ).strip().lower()
    if configured_backend not in {"auto", "mongodb", "inmemory"}:
        raise ValueError("local_workspace_document_store_backend_invalid")

    mongo_uri = (os.environ.get("INTERGRAX_MONGODB_URI") or "").strip()
    if configured_backend == "mongodb" or (
        configured_backend == "auto" and mongo_uri
    ):
        if not mongo_uri:
            raise RuntimeError("lkw_durable_store_configuration_missing")
        from intergrax.integrations.providers.document_store.mongodb.bundle import (
            create_mongodb_document_store,
        )

        collection_name = (
            os.environ.get("LKW_MANAGED_WORKSPACE_COLLECTION", _DEFAULT_COLLECTION).strip()
            or _DEFAULT_COLLECTION
        )
        return create_mongodb_document_store(collection_name=collection_name)

    if configured_backend == "mongodb":
        raise RuntimeError("lkw_durable_store_configuration_missing")

    return InMemoryDocumentStore()
