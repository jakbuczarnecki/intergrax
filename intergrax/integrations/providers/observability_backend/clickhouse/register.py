# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register clickhouse in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend
from intergrax.integrations.providers.observability_backend.clickhouse.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.clickhouse.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_clickhouse_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_clickhouse_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
