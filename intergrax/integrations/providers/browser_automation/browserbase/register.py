# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register browserbase in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.browser_automation.browserbase.bundle import create_browserbase_browser_automation
from intergrax.integrations.providers.browser_automation.browserbase.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.browser_automation.browserbase.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_browserbase_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_browserbase_browser_automation,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
