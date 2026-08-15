"""Provider-owned helpers for building Vendor Knowledge contributions."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

AdapterRegistrar = Callable[[KnowledgeAdapterRegistry], object]


def build_adapter(registrar: AdapterRegistrar) -> object:
    """Instantiate one existing adapter through its provider-owned registrar."""
    registry = KnowledgeAdapterRegistry()
    adapter = registrar(registry)
    if len(registry.registered_keys()) != 1:
        raise ValueError("vendor_knowledge_adapter_builder_invalid")
    return adapter


def build_durable_source_plugin(
    *,
    provider_id: str,
    integration_category: IntegrationCategory,
    source_kind: str,
    runtime_ref: str,
    indexed_runtime_ref: str | None = None,
) -> VendorKnowledgeSourcePlugin:
    capabilities = [
        VendorKnowledgeModeCapability(
            mode=VendorKnowledgeMode.DURABLE,
            contract_version="vendor-knowledge.durable.v1",
            operations=("inventory", "snapshot", "reconciliation", "exact_fetch"),
            runtime_ref=runtime_ref,
            constraints={"application_sink": "platform_foundation"},
        )
    ]
    if indexed_runtime_ref is not None:
        capabilities.append(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("eligible", "materialize", "publish", "index"),
                runtime_ref=indexed_runtime_ref,
                constraints={"application_proof": "vk4"},
            )
        )
    return VendorKnowledgeSourcePlugin(
        identity=VendorKnowledgeSourceIdentity(
            provider_id=provider_id,
            integration_category=integration_category,
            source_kind=source_kind,
        ),
        capabilities=tuple(capabilities),
    )


__all__ = [
    "AdapterRegistrar",
    "build_adapter",
    "build_durable_source_plugin",
]
