# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register hubspot in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.crm.hubspot.bundle import create_hubspot_crm
from intergrax.integrations.providers.crm.hubspot.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_hubspot_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_hubspot_crm, override=override)
