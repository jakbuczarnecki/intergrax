# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register newrelic in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.newrelic.bundle import create_newrelic_observability_backend
from intergrax.integrations.providers.observability_backend.newrelic.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.newrelic.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_newrelic_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_newrelic_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
