# © Artur Czarnecki. All rights reserved.

"""Verify managed-workspace logical documents still have vector-store records."""

from __future__ import annotations

from typing import Any

from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import (
    create_vectorstore_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.scope import vectorstore_tenant_id


class ManagedWorkspaceIndexedVectorVerifier:
    """Checks vector presence for one logical managed-workspace document."""

    def __init__(
        self,
        vectorstore_manager: VectorstoreManager,
        *,
        tenant_vectorstore_cache: dict[str, Any] | None = None,
        integration_profile: Any | None = None,
    ) -> None:
        self._default_manager = vectorstore_manager
        self._tenant_vectorstore_cache = tenant_vectorstore_cache or {}
        self._integration_profile = integration_profile

    def _resolve_manager(self, tenant_id: str) -> VectorstoreManager:
        cached = self._tenant_vectorstore_cache.get(tenant_id)
        if cached is not None:
            return cached
        wired_tenant = vectorstore_tenant_id(self._default_manager)
        if wired_tenant is not None and wired_tenant == tenant_id:
            return self._default_manager
        if self._integration_profile is not None:
            return create_vectorstore_manager(
                tenant_id=tenant_id,
                profile=self._integration_profile,
            )
        return self._default_manager

    def has_indexed_vectors(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        document_id: str,
    ) -> bool:
        try:
            vector_ids = self._resolve_manager(tenant_id).list_source_record_ids(
                source_id=source_id,
                root_document_id=document_id,
                scope=VectorStoreScope(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                ),
            )
        except (RuntimeError, TypeError, ValueError):
            return False
        return bool(vector_ids)
