"""Databricks Vendor Knowledge connection contribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
)
from intergrax.integrations.providers.relational_store.databricks.tenant_connection_factory import (
    DatabricksTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeProviderContribution,
)


def build_databricks_vendor_knowledge_contribution(
    *,
    connection_factory: Callable[[], Any] | None = None,
) -> VendorKnowledgeProviderContribution:
    return VendorKnowledgeProviderContribution(
        provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
                integration_category=IntegrationCategory.RELATIONAL_STORE,
                factory=DatabricksTenantConnectionIntegrationFactory(
                    connection_factory=connection_factory,
                ),
            ),
        ),
    )


__all__ = ["build_databricks_vendor_knowledge_contribution"]
