# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register auth0 in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.identity_provider.auth0.bundle import create_auth0_identity_provider
from intergrax.integrations.providers.identity_provider.auth0.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.identity_provider.auth0.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_auth0_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_auth0_identity_provider,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
