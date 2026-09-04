# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register doppler in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.doppler.bundle import create_doppler_secrets_store
from intergrax.integrations.providers.secrets_store.doppler.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.secrets_store.doppler.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_doppler_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_doppler_secrets_store,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
