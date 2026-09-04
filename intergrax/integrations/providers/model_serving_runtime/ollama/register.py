# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register ollama in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.model_serving_runtime.ollama.bundle import create_ollama_model_serving_runtime
from intergrax.integrations.providers.model_serving_runtime.ollama.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.model_serving_runtime.ollama.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_ollama_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_ollama_model_serving_runtime,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
