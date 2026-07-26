# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Instance-local registry for vendor knowledge source adapters."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contracts import VendorKnowledgeAdapter
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceRef

type AdapterRegistryKey = tuple[str, IntegrationCategory, str]


class KnowledgeAdapterRegistry:
    """Explicit registration of thin source adapters keyed by provider identity.

    Instance-local only — not a catalog, singleton, or import-time plugin system.
    """

    def __init__(self) -> None:
        self._adapters: dict[AdapterRegistryKey, VendorKnowledgeAdapter] = {}

    def register(self, adapter: VendorKnowledgeAdapter) -> None:
        provider_id = str(adapter.provider_id).strip()
        if not provider_id:
            raise ValueError("provider_id must be a non-empty string")
        source_kind = str(adapter.source_kind).strip()
        if not source_kind:
            raise ValueError("source_kind must be a non-empty string")
        integration_kind = adapter.integration_kind
        if not isinstance(integration_kind, IntegrationCategory):
            raise ValueError("integration_kind must be an IntegrationCategory")

        key: AdapterRegistryKey = (provider_id, integration_kind, source_kind)
        if key in self._adapters:
            raise ValueError(
                "adapter already registered for "
                f"({provider_id!r}, {integration_kind.value!r}, {source_kind!r})"
            )
        self._adapters[key] = adapter

    def resolve(self, *, source: KnowledgeSourceRef) -> VendorKnowledgeAdapter:
        key: AdapterRegistryKey = (
            source.provider_id,
            source.integration_kind,
            source.source_kind,
        )
        adapter = self._adapters.get(key)
        if adapter is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND,
                safe_message="No knowledge source adapter is registered for the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return adapter

    def registered_keys(self) -> tuple[AdapterRegistryKey, ...]:
        return tuple(self._adapters.keys())
