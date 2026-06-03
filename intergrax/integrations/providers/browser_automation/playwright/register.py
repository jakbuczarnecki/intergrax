# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register playwright in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation
from intergrax.integrations.providers.browser_automation.playwright.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_playwright_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_playwright_browser_automation, override=override)
