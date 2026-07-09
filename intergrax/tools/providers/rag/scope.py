# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tenant scope helpers for RAG ingest/retrieve tool paths."""

from __future__ import annotations

from typing import Any

from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import create_vectorstore_manager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
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


def _tenant_id_from_config(config: object | None) -> str | None:
    if config is None:
        return None
    tenant = attribute_access.optional(config, "tenant_id", None)
    if tenant is not None and str(tenant).strip():
        return str(tenant).strip()
    return None


def vectorstore_tenant_id(manager: object | None) -> str | None:
    if manager is None:
        return None

    store: object | None = attribute_access.optional(manager, "_store", manager)
    seen: set[int] = set()
    while store is not None and id(store) not in seen:
        seen.add(id(store))

        for config_attr in ("cfg", "store_config", "_store_config", "_config", "config"):
            tenant = _tenant_id_from_config(attribute_access.optional(store, config_attr, None))
            if tenant is not None:
                return tenant

        inner = attribute_access.optional(store, "_inner", None)
        if inner is None:
            inner = attribute_access.optional(store, "rag_store", None)
        if inner is None or inner is store:
            break
        store = inner

    tenant = attribute_access.optional(store, "_tenant_id", None)
    if tenant is not None and str(tenant).strip():
        return str(tenant).strip()
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
    if manager is None or not tenant_id:
        return manager

    wired_tenant = vectorstore_tenant_id(manager) or "default"
    if wired_tenant == tenant_id:
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
        return cached

    scoped = create_vectorstore_manager(tenant_id=tenant_id, profile=profile)
    cache_obj[tenant_id] = scoped
    return scoped
