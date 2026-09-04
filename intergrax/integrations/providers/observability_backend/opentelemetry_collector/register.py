# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register opentelemetry_collector in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.opentelemetry_collector.bundle import create_opentelemetry_collector_observability_backend
from intergrax.integrations.providers.observability_backend.opentelemetry_collector.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.opentelemetry_collector.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_opentelemetry_collector_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_opentelemetry_collector_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
