# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tenant scope helpers for RAG ingest/retrieve tool paths."""

from __future__ import annotations

from typing import Any

from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import create_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.tools.providers.rag.source_operation_wiring import bind_source_operation_coordinator
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.utils import attribute_access

TENANT_ID_METADATA_CONFLICT = "tenant_id_metadata_conflict"
_TENANT_VECTORSTORE_CACHE_KEY = "tenant_vectorstore_managers"


def authoritative_tenant_id(
    *,
    request_tenant: str | None,
    metadata_tenant: Any = None,
) -> tuple[str | None, str | None]:
    """Resolve tenant_id; metadata must not conflict with authoritative request tenant."""
    auth = str(request_tenant).strip() if request_tenant is not None and str(request_tenant).strip() else None
    meta = (
        str(metadata_tenant).strip()
        if metadata_tenant is not None and str(metadata_tenant).strip()
        else None
    )
    if auth is not None:
        if meta is not None and meta != auth:
            return None, TENANT_ID_METADATA_CONFLICT
        return auth, None
    return meta, None


def vectorstore_tenant_id(manager: object | None) -> str | None:
    if manager is None:
        return None

    bound_scope = attribute_access.optional(manager, "bound_scope", None)
    if isinstance(bound_scope, VectorStoreScope):
        return bound_scope.tenant_id
    return None


def use_wired_retrieval_managers(
    ctx: ToolWiringContext,
    vectorstore: BaseVectorstoreManager | None,
) -> bool:
    """Return True when wired retriever/retrieval service targets the same store tenant."""
    if vectorstore is None:
        return True
    wired = ctx.vectorstore_manager
    if wired is None or vectorstore is wired:
        return True
    wired_tenant = vectorstore_tenant_id(wired)
    scoped_tenant = vectorstore_tenant_id(vectorstore)
    return wired_tenant is not None and wired_tenant == scoped_tenant


def resolve_tenant_scoped_vectorstore(
    ctx: ToolWiringContext,
    tenant_id: str | None,
) -> BaseVectorstoreManager | None:
    """Return a vectorstore manager whose provider tenant matches ``tenant_id`` when set."""
    manager = ctx.vectorstore_manager
    if not tenant_id:
        return None

    if manager is not None:
        wired_tenant = vectorstore_tenant_id(manager)
        if wired_tenant is not None and wired_tenant == tenant_id:
            bind_source_operation_coordinator(ctx, manager)
            return manager

    profile = ctx.integration_profile
    if profile is None:
        return manager

    cache_obj = ctx.extras.get(_TENANT_VECTORSTORE_CACHE_KEY)
    if not isinstance(cache_obj, dict):
        cache_obj = {}
        ctx.extras[_TENANT_VECTORSTORE_CACHE_KEY] = cache_obj

    cached = cache_obj.get(tenant_id)
    if cached is not None:
        bind_source_operation_coordinator(ctx, cached)
        return cached

    scoped = create_vectorstore_manager(tenant_id=tenant_id, profile=profile)
    bind_source_operation_coordinator(ctx, scoped)
    cache_obj[tenant_id] = scoped
    return scoped
