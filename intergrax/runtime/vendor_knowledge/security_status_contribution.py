# © Artur Czarnecki. All rights reserved.

"""Security Status Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.security_status.config import SecurityStatusIntegrationConfig
from intergrax.integrations.providers.security_status.knowledge_read import (
    SECURITY_STATUS_PROVIDER_ID,
)
from intergrax.integrations.providers.security_status.tenant_connection_factory import (
    SecurityStatusTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.security_status.registration import (
    build_security_status_live_registration_bundles,
    build_security_status_vendor_knowledge_source_plugin,
)


def build_security_status_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[SecurityStatusIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.SECURITY_SCANNER
    return VendorKnowledgeProviderContribution(
        provider_id=SECURITY_STATUS_PROVIDER_ID,
        integration_category=category,
        source_plugins=(build_security_status_vendor_knowledge_source_plugin(),),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=SECURITY_STATUS_PROVIDER_ID,
                integration_category=category,
                factory=SecurityStatusTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
        live_contributions=build_security_status_live_registration_bundles(),
    )


__all__ = ["build_security_status_vendor_knowledge_contribution"]
