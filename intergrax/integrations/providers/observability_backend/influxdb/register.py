# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register influxdb in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.influxdb.bundle import create_influxdb_observability_backend
from intergrax.integrations.providers.observability_backend.influxdb.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.influxdb.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_influxdb_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_influxdb_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
