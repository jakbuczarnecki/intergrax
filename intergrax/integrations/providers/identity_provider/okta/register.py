# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register okta in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.okta.bundle import create_okta_identity_provider
from intergrax.integrations.providers.identity_provider.okta.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.identity_provider.okta.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_okta_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_okta_identity_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
