# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register vault in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.vault.bundle import create_vault_secrets_store
from intergrax.integrations.providers.secrets_store.vault.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_vault_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_vault_secrets_store, override=override)
