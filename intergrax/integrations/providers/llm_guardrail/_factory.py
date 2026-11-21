# © Artur Czarnecki. All rights reserved.

"""Guardrail slug → backend factory router (M-P12)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
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
from intergrax.integrations.providers.llm_guardrail._vendor_opens import vendor_engine_ready
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

_VENDOR_PROBE_SLUGS = frozenset(_GUARD_FACTORIES.keys())


def _pattern_fallback_allowed(slug: str, lab_opt_in: LabSelectorOptIn | None) -> bool:
    if lab_opt_in is None:
        return False
    return lab_opt_in.selector.strip() == slug.strip()


def _lab_opt_in_for_slug(slug: str, lab_opt_in: LabSelectorOptIn | None) -> LabSelectorOptIn | None:
    if lab_opt_in is None:
        return None
    if _pattern_fallback_allowed(slug, lab_opt_in):
        return lab_opt_in
    return LabSelectorOptIn(
        selector=slug,
        reason=f"laboratory guardrail pattern fallback companion for chained slug {slug!r}",
    )


def _require_vendor_engine(slug: str) -> None:
    if slug not in _VENDOR_PROBE_SLUGS:
        return
    if not vendor_engine_ready(slug):
        raise IntegrationConfigurationError(f"{slug!r} vendor engine is not available")


def create_guardrail_backend(
    slug: str,
    *,
    provider_options: Mapping[str, Any] | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> LlmGuardrailBackend:
    allow_pattern = _pattern_fallback_allowed(slug, lab_opt_in)
    factory = _GUARD_FACTORIES.get(slug)
    if factory is None:
        if lab_opt_in is None or lab_opt_in.selector.strip() != slug.strip():
            raise IntegrationConfigurationError(f"unknown guardrail slug {slug!r}")
        return create_stub_guardrail(slug)
    if not allow_pattern:
        _require_vendor_engine(slug)
    factory_kwargs: dict[str, Any] = {"allow_pattern_fallback": allow_pattern}
    if slug in _PROVIDER_OPTIONS_SLUGS:
        factory_kwargs["provider_options"] = provider_options
        return factory(**factory_kwargs)
    return factory(**factory_kwargs)


def create_chained_guardrail_backend(
    *slugs: str,
    provider_options_map: Mapping[str, Mapping[str, Any]] | None = None,
    lab_opt_in: LabSelectorOptIn | None = None,
) -> LlmGuardrailBackend:
    opts_map = provider_options_map or {}
    backends = [
        create_guardrail_backend(
            slug,
            provider_options=opts_map.get(slug),
            lab_opt_in=_lab_opt_in_for_slug(slug, lab_opt_in),
        )
        for slug in slugs
        if slug
    ]
    if not backends:
        raise ValueError("create_chained_guardrail_backend requires at least one slug")
    if len(backends) == 1:
        return backends[0]
    return ChainedGuardrailBackend(*backends)
