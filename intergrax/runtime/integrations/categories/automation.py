# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Automation, billing, and CRM provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_READ_HEALTH,
    _CONNECT_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

BROWSER_AUTOMATION_INTEGRATION_CONTRACT_SCHEMA = "browser_automation_integration_contract.v1"
BILLING_METER_INTEGRATION_CONTRACT_SCHEMA = "billing_meter_integration_contract.v1"
CRM_INTEGRATION_CONTRACT_SCHEMA = "crm_integration_contract.v1"


class BrowserAutomationIntegrationContract(PlatformIntegrationContract):
    """Category contract for browser_automation providers (playwright, selenium, …)."""

    schema_id: Literal["browser_automation_integration_contract.v1"] = (
        BROWSER_AUTOMATION_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.BROWSER_AUTOMATION.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> BrowserAutomationIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.BROWSER_AUTOMATION.value,
            default_capabilities=_CONNECT_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class BillingMeterIntegrationContract(PlatformIntegrationContract):
    """Category contract for billing_meter providers (stripe, …)."""

    schema_id: Literal["billing_meter_integration_contract.v1"] = BILLING_METER_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.BILLING_METER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> BillingMeterIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.BILLING_METER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class CrmIntegrationContract(PlatformIntegrationContract):
    """Category contract for crm providers (salesforce, hubspot, …)."""

    schema_id: Literal["crm_integration_contract.v1"] = CRM_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.CRM.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> CrmIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.CRM.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )
