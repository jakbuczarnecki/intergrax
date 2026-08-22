# © Artur Czarnecki. All rights reserved.

"""Guardrail slug → backend factory router (M-P12)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
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

_PROVIDER_OPTIONS_SLUGS = frozenset({"bedrock_guardrails", "nemo_guardrails", "llama_guard"})


def create_guardrail_backend(
    slug: str,
    *,
    provider_options: Mapping[str, Any] | None = None,
) -> LlmGuardrailBackend:
    factory = _GUARD_FACTORIES.get(slug)
    if factory is None:
        return create_stub_guardrail(slug)
    if slug in _PROVIDER_OPTIONS_SLUGS:
        return factory(provider_options=provider_options)
    return factory()


def create_chained_guardrail_backend(
    *slugs: str,
    provider_options_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> LlmGuardrailBackend:
    opts_map = provider_options_map or {}
    backends = [
        create_guardrail_backend(slug, provider_options=opts_map.get(slug))
        for slug in slugs
        if slug
    ]
    if not backends:
        raise ValueError("create_chained_guardrail_backend requires at least one slug")
    if len(backends) == 1:
        return backends[0]
    return ChainedGuardrailBackend(*backends)
