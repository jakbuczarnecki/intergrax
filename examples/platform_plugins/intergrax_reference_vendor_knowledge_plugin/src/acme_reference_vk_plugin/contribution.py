"""Vendor Knowledge provider contribution entry point."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeDiscoveryContribution,
    VendorKnowledgeIndexedMaterializerContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_builder import build_durable_source_plugin
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity

from acme_reference_vk_plugin.adapter import AcmeReferenceDocumentsKnowledgeAdapter
from acme_reference_vk_plugin.constants import (
    ACME_ADAPTER_RUNTIME_REF,
    ACME_DOCUMENTS_SOURCE_KIND,
    ACME_INDEXED_RUNTIME_REF,
    ACME_REFERENCE_PROVIDER_ID,
)
from acme_reference_vk_plugin.discovery import build_acme_reference_discovery_strategy
from acme_reference_vk_plugin.factory import AcmeReferenceTenantConnectionIntegrationFactory
from acme_reference_vk_plugin.materializer import AcmeReferenceDocumentMaterializer


def build_acme_reference_contribution() -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.WIKI_KNOWLEDGE
    identity = VendorKnowledgeSourceIdentity(
        provider_id=ACME_REFERENCE_PROVIDER_ID,
        integration_category=category,
        source_kind=ACME_DOCUMENTS_SOURCE_KIND,
    )
    return VendorKnowledgeProviderContribution(
        provider_id=ACME_REFERENCE_PROVIDER_ID,
        integration_category=category,
        adapters=(AcmeReferenceDocumentsKnowledgeAdapter(),),
        source_plugins=(
            build_durable_source_plugin(
                provider_id=ACME_REFERENCE_PROVIDER_ID,
                integration_category=category,
                source_kind=ACME_DOCUMENTS_SOURCE_KIND,
                runtime_ref=ACME_ADAPTER_RUNTIME_REF,
                indexed_runtime_ref=ACME_INDEXED_RUNTIME_REF,
            ),
        ),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=ACME_REFERENCE_PROVIDER_ID,
                integration_category=category,
                factory=AcmeReferenceTenantConnectionIntegrationFactory(),
            ),
        ),
        discovery_contributions=(
            VendorKnowledgeDiscoveryContribution(
                identity=identity,
                factory=build_acme_reference_discovery_strategy,
            ),
        ),
        indexed_materializers=(
            VendorKnowledgeIndexedMaterializerContribution(
                identity=identity,
                runtime_ref=ACME_INDEXED_RUNTIME_REF,
                factory=AcmeReferenceDocumentMaterializer,
            ),
        ),
    )
