# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Single context-window resolution path for all LLM adapters (ADR-LLM-002)."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.catalog_miss_diag import (
    CatalogResolutionTier,
    maybe_emit_catalog_miss,
)
from intergrax.llm_adapters.registry.gateway_metadata.session import lookup_gateway_context_window
from intergrax.llm_adapters.registry.model_catalog import ModelCatalog, get_model_catalog


def _provider_slug(provider: LLMProvider | str) -> str:
    if isinstance(provider, LLMProvider):
        return provider.value
    return str(provider or "").strip().lower()


def pop_context_window_override(kwargs: dict[str, Any]) -> int | None:
    """Extract operator override from adapter constructor kwargs."""
    raw = kwargs.pop("context_window_tokens", None)
    if raw is None:
        return None
    value = int(raw)
    return value if value > 0 else None


def _record_catalog_miss(
    provider: LLMProvider | str,
    model: str,
    resolved_tokens: int,
    *,
    resolution_tier: CatalogResolutionTier,
    run_id: str | None,
    profile_options: Mapping[str, Any],
) -> None:
    maybe_emit_catalog_miss(
        provider,
        model,
        resolved_tokens,
        resolution_tier=resolution_tier,
        run_id=run_id or profile_options.get("run_id"),  # type: ignore[arg-type]
    )


def resolve_context_window_tokens(
    provider: LLMProvider | str,
    model: str,
    *,
    profile_options: Mapping[str, Any] | None = None,
    legacy_windows: Mapping[str, int] | None = None,
    catalog: ModelCatalog | None = None,
    run_id: str | None = None,
) -> int:
    """
    Deterministic resolution order (ADR-LLM-002):

    1. ``profile_options["context_window_tokens"]`` or ctor override
    2. ModelCatalog exact match — no miss diagnostic
    3. ModelCatalog prefix rules — ``ModelCatalogMissDiagV1`` (``prefix_rule``)
    4. Optional gateway metadata session merge (``fetch_gateway_metadata``)
    5. Legacy per-adapter dict (deprecated)
    6. Provider family default — miss diagnostic (``provider_default``)
    7. Catalog fallback_default — miss diagnostic (``fallback_default``)
    """
    options = dict(profile_options or {})
    override = options.get("context_window_tokens")
    if override is not None and int(override) > 0:
        return int(override)

    normalized_model = (model or "").strip()
    cat = catalog or get_model_catalog()
    effective_run_id = run_id or options.get("run_id")  # type: ignore[assignment]

    exact = cat.lookup_exact(normalized_model)
    if exact is not None:
        return exact.context_window_tokens

    prefix_tokens = cat.lookup_prefix(normalized_model)
    if prefix_tokens is not None:
        _record_catalog_miss(
            provider,
            normalized_model,
            prefix_tokens,
            resolution_tier=CatalogResolutionTier.PREFIX_RULE,
            run_id=effective_run_id,
            profile_options=options,
        )
        return prefix_tokens

    gateway_tokens = lookup_gateway_context_window(provider, normalized_model, options)
    if gateway_tokens is not None:
        return gateway_tokens

    if legacy_windows is not None:
        legacy_hit = legacy_windows.get(normalized_model)
        if legacy_hit is not None:
            return int(legacy_hit)

    provider_default = cat.provider_default(_provider_slug(provider))
    if provider_default is not None:
        _record_catalog_miss(
            provider,
            normalized_model,
            provider_default,
            resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT,
            run_id=effective_run_id,
            profile_options=options,
        )
        return provider_default

    fallback = int(cat.fallback_default)
    _record_catalog_miss(
        provider,
        normalized_model,
        fallback,
        resolution_tier=CatalogResolutionTier.FALLBACK_DEFAULT,
        run_id=effective_run_id,
        profile_options=options,
    )
    return fallback


def init_adapter_context_window_tokens(
    *,
    provider: LLMProvider | str,
    model: str,
    constructor_kwargs: dict[str, Any],
    legacy_windows: Mapping[str, int] | None = None,
    catalog: ModelCatalog | None = None,
) -> int:
    """Pop override from ctor kwargs and resolve context window for adapter ``__init__``."""
    override = pop_context_window_override(constructor_kwargs)
    profile_options = {"context_window_tokens": override} if override is not None else None
    return resolve_context_window_tokens(
        provider,
        model,
        profile_options=profile_options,
        legacy_windows=legacy_windows,
        catalog=catalog,
    )
