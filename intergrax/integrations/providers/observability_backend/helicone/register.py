# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register helicone in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.helicone.bundle import create_helicone_observability_backend
from intergrax.integrations.providers.observability_backend.helicone.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.helicone.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_helicone_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_helicone_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
