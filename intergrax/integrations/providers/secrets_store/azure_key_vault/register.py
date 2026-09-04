# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register azure_key_vault in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.azure_key_vault.bundle import create_azure_key_vault_secrets_store
from intergrax.integrations.providers.secrets_store.azure_key_vault.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.secrets_store.azure_key_vault.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_azure_key_vault_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_azure_key_vault_secrets_store,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
