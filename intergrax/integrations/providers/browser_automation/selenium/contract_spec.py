# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Selenium browser automation."""

from __future__ import annotations

from intergrax.integrations.providers.browser_automation.selenium.bundle import (
    create_selenium_browser_automation_integration,
)
from intergrax.integrations.providers.browser_automation.selenium.integration import (
    SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
    SeleniumBrowserAutomationIntegration,
    SeleniumBrowserAutomationIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="browser_automation",
    provider_id=SELENIUM_BROWSER_AUTOMATION_PROVIDER_ID,
    integration_class=SeleniumBrowserAutomationIntegration,
    contract_class=BrowserAutomationIntegrationContract,
    contract_factory=create_selenium_browser_automation_integration,
    display_name="Selenium",
    config_class=SeleniumBrowserAutomationIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
