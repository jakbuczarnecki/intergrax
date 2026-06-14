# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Single context-window resolution path for all LLM adapters (ADR-LLM-002)."""

from __future__ import annotations

from typing import Any, Mapping

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
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


def resolve_context_window_tokens(
    provider: LLMProvider | str,
    model: str,
    *,
    profile_options: Mapping[str, Any] | None = None,
    legacy_windows: Mapping[str, int] | None = None,
    catalog: ModelCatalog | None = None,
) -> int:
    """
    Deterministic resolution order (ADR-LLM-002):

    1. ``profile_options["context_window_tokens"]`` or ctor override
    2. ModelCatalog exact match
    3. ModelCatalog prefix rules
    4. Legacy per-adapter dict (deprecated)
    5. Provider family default from catalog
    6. Catalog fallback_default
    """
    options = dict(profile_options or {})
    override = options.get("context_window_tokens")
    if override is not None and int(override) > 0:
        return int(override)

    normalized_model = (model or "").strip()
    cat = catalog or get_model_catalog()

    exact = cat.lookup_exact(normalized_model)
    if exact is not None:
        return exact.context_window_tokens

    prefix_tokens = cat.lookup_prefix(normalized_model)
    if prefix_tokens is not None:
        return prefix_tokens

    if legacy_windows is not None:
        legacy_hit = legacy_windows.get(normalized_model)
        if legacy_hit is not None:
            return int(legacy_hit)

    provider_default = cat.provider_default(_provider_slug(provider))
    if provider_default is not None:
        return provider_default

    return int(cat.fallback_default)


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
