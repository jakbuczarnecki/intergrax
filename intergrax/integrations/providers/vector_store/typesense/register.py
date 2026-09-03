# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register typesense in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.typesense.bundle import create_typesense_vector_store
from intergrax.integrations.providers.vector_store.typesense.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.vector_store.typesense.contract_spec import CONTRACT_SPECS


def register_typesense_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_typesense_vector_store, override=override, contract_specs=CONTRACT_SPECS)
