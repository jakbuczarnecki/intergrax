# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register replicate in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.ml_inference_host.replicate.bundle import create_replicate_ml_inference_host
from intergrax.integrations.providers.ml_inference_host.replicate.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.ml_inference_host.replicate.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_replicate_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_replicate_ml_inference_host,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
