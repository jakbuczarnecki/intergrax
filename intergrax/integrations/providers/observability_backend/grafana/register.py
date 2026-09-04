# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register grafana in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.grafana.bundle import create_grafana_observability_backend
from intergrax.integrations.providers.observability_backend.grafana.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.grafana.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_grafana_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_grafana_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
