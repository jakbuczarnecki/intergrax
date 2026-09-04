# © Artur Czarnecki. All rights reserved.

"""Factory helpers for NeMo Guardrails guardrail adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails import create_nemo_guardrails_backend
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail.nemo_guardrails.integration import (
    NEMO_GUARDRAILS_PROVIDER_ID,
    NemoGuardrailsLlmGuardrailIntegration,
    NemoGuardrailsLlmGuardrailIntegrationConfig,
)

__all__ = [
    "create_nemo_guardrails_llm_guardrail",
    "create_nemo_guardrails_llm_guardrail_integration",
]


def create_nemo_guardrails_llm_guardrail_integration(
    *,
    backend: LlmGuardrailBackend | None = None,
    enabled: bool = False,
) -> NemoGuardrailsLlmGuardrailIntegration:
    """Build a contract-based NeMo Guardrails guardrail integration."""
    if backend is not None:
        return NemoGuardrailsLlmGuardrailIntegration.from_backend(
            backend,
            provider_id=NEMO_GUARDRAILS_PROVIDER_ID,
            display_name="NeMo Guardrails",
            enabled=enabled,
            config=NemoGuardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
        )
    return NemoGuardrailsLlmGuardrailIntegration.for_provider(
        provider_id=NEMO_GUARDRAILS_PROVIDER_ID,
        display_name="NeMo Guardrails",
        config=NemoGuardrailsLlmGuardrailIntegrationConfig(enabled=enabled),
    )


def create_nemo_guardrails_llm_guardrail(
    *,
    provider_options: Mapping[str, Any] | None = None,
    **kwargs: object,
) -> NemoGuardrailsLlmGuardrailIntegration:
    """Catalog factory for ``nemo_guardrails`` / ``llm_guardrail``."""
    opts = provider_options
    if opts is None and kwargs:
        opts = dict(kwargs)
    backend = create_nemo_guardrails_backend(provider_options=opts)
    return NemoGuardrailsLlmGuardrailIntegration.from_backend(
        backend,
        provider_id=NEMO_GUARDRAILS_PROVIDER_ID,
        display_name="NeMo Guardrails",
        enabled=True,
        config=NemoGuardrailsLlmGuardrailIntegrationConfig(enabled=True),
    )
