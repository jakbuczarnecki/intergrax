# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register arize in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.arize.bundle import create_arize_observability_backend
from intergrax.integrations.providers.observability_backend.arize.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.arize.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_arize_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_arize_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
