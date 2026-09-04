# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.integrations.providers.graph_store.neptune.bundle import create_neptune_graph_store
from intergrax.integrations.providers.graph_store.neptune.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.graph_store.neptune.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_neptune_integration(*, override: bool = False) -> None:
    register_from_manifest(MANIFEST, create_neptune_graph_store, override=override, contract_specs=CONTRACT_SPECS)
