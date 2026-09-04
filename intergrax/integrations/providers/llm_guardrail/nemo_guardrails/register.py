# © Artur Czarnecki. All rights reserved.

"""Register nemo_guardrails in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.bundle import create_nemo_guardrails_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_nemo_guardrails_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_nemo_guardrails_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
