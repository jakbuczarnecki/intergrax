# © Artur Czarnecki. All rights reserved.

"""Register openguardrails in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.openguardrails.bundle import create_openguardrails_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.openguardrails.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.openguardrails.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_openguardrails_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_openguardrails_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
