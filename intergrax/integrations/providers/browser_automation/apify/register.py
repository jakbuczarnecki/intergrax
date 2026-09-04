# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register apify in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.browser_automation.apify.bundle import create_apify_browser_automation
from intergrax.integrations.providers.browser_automation.apify.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.browser_automation.apify.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_apify_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_apify_browser_automation,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
