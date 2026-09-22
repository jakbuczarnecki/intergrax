# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability discovery for LLM provider cancellation (W4-D)."""

from __future__ import annotations

from intergrax.contracts.external_operation_termination import ExternalOperationCapabilities


def external_operation_capabilities_for_provider(
    provider_slug: str,
) -> ExternalOperationCapabilities:
    from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

    return LLMAdapterRegistry.external_operation_capabilities_for(provider_slug)
