# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register langfuse in the integration catalog.

Registry hook only — uses the legacy query facade factory. Contract-based
``LangfuseObservabilityIntegration`` registration (registry v2 / runtime contract
catalog) remains deferred until a safe registry mechanism exists.
"""

from __future__ import annotations

from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend
from intergrax.integrations.providers.observability_backend.langfuse.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.observability_backend.langfuse.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_langfuse_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_langfuse_observability_backend,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
