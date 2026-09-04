# © Artur Czarnecki. All rights reserved.

"""Register lakera in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.lakera.bundle import create_lakera_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.lakera.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.lakera.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_lakera_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_lakera_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
