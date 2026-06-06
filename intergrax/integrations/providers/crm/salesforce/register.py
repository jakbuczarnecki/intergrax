# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register salesforce in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.crm.salesforce.bundle import create_salesforce_crm
from intergrax.integrations.providers.crm.salesforce.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_salesforce_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_salesforce_crm, override=override)
