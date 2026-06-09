# © Artur Czarnecki. All rights reserved.

"""Guardrail slug → backend factory router (M-P12)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.integrations.contracts.llm_guardrail import GuardrailBackendOptions, LlmGuardrailBackend
from intergrax.integrations.providers.llm_guardrail._adapters import (
    create_azure_content_safety_backend,
    create_bedrock_guardrails_backend,
    create_guardrails_ai_backend,
    create_lakera_backend,
    create_llama_guard_backend,
    create_llm_guard_backend,
    create_nemo_guardrails_backend,
    create_openguardrails_backend,
    create_presidio_backend,
)
from intergrax.integrations.providers.llm_guardrail._stub_backend import create_stub_guardrail
from intergrax.integrations.providers.llm_guardrail.bundles.chained import ChainedGuardrailBackend

_GUARD_FACTORIES: dict[str, Callable[..., LlmGuardrailBackend]] = {
    "llm_guard": create_llm_guard_backend,
    "guardrails_ai": create_guardrails_ai_backend,
    "nemo_guardrails": create_nemo_guardrails_backend,
    "openguardrails": create_openguardrails_backend,
    "presidio": create_presidio_backend,
    "llama_guard": create_llama_guard_backend,
    "lakera": create_lakera_backend,
    "azure_content_safety": create_azure_content_safety_backend,
    "bedrock_guardrails": create_bedrock_guardrails_backend,
}


def create_guardrail_backend(
    slug: str,
    *,
    options: GuardrailBackendOptions | None = None,
) -> LlmGuardrailBackend:
    factory = _GUARD_FACTORIES.get(slug)
    if factory is None:
        return create_stub_guardrail(slug)
    return factory(options=options)


def create_chained_guardrail_backend(
    *slugs: str,
    options: GuardrailBackendOptions | None = None,
) -> LlmGuardrailBackend:
    backends = [create_guardrail_backend(slug, options=options) for slug in slugs if slug]
    if not backends:
        raise ValueError("create_chained_guardrail_backend requires at least one slug")
    if len(backends) == 1:
        return backends[0]
    return ChainedGuardrailBackend(*backends)
