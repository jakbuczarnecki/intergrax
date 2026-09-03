# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register pgvector in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store
from intergrax.integrations.providers.vector_store.pgvector.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.integrations.providers.vector_store.pgvector.contract_spec import CONTRACT_SPECS


def register_pgvector_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_pgvector_vector_store, override=override, contract_specs=CONTRACT_SPECS)
