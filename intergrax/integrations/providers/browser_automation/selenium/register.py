# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register selenium in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation
from intergrax.integrations.providers.browser_automation.selenium.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_selenium_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_selenium_browser_automation, override=override)
