# © Artur Czarnecki. All rights reserved.

"""Register llm_guard in the integration catalog."""

from __future__ import annotations

from intergrax.integrations.providers.llm_guardrail.llm_guard.bundle import create_llm_guard_llm_guardrail
from intergrax.integrations.providers.llm_guardrail.llm_guard.contract_spec import CONTRACT_SPECS
from intergrax.integrations.providers.llm_guardrail.llm_guard.manifest import MANIFEST
from intergrax.integrations.registry.plugin_register import register_from_manifest


def register_llm_guard_integration(*, override: bool = False) -> None:
    register_from_manifest(
        MANIFEST,
        create_llm_guard_llm_guardrail,
        override=override,
        contract_specs=CONTRACT_SPECS,
    )
