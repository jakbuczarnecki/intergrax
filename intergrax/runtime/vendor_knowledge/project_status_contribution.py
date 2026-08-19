# © Artur Czarnecki. All rights reserved.

"""Project Status Vendor Knowledge contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.integrations.providers.project_status.tenant_connection_factory import (
    ProjectStatusTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.project_status.registration import (
    build_project_status_live_registration_bundles,
    build_project_status_vendor_knowledge_source_plugin,
)


def build_project_status_vendor_knowledge_contribution(
    *,
    http_client_factory: Callable[[ProjectStatusIntegrationConfig], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    category = IntegrationCategory.ISSUE_TRACKER
    return VendorKnowledgeProviderContribution(
        provider_id=PROJECT_STATUS_PROVIDER_ID,
        integration_category=category,
        source_plugins=(build_project_status_vendor_knowledge_source_plugin(),),
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=PROJECT_STATUS_PROVIDER_ID,
                integration_category=category,
                factory=ProjectStatusTenantConnectionIntegrationFactory(
                    http_client_factory=http_client_factory,
                ),
            ),
        ),
        live_contributions=build_project_status_live_registration_bundles(),
    )


__all__ = ["build_project_status_vendor_knowledge_contribution"]
