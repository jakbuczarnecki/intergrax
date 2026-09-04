# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register gcp_secret_manager in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.gcp_secret_manager.bundle import create_gcp_secret_manager_secrets_store
from intergrax.integrations.providers.secrets_store.gcp_secret_manager.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.secrets_store.gcp_secret_manager.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_gcp_secret_manager_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_gcp_secret_manager_secrets_store,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
