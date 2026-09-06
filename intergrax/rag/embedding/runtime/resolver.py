# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Canonical embedding provider runtime resolver (IntegrationProfile → EmbeddingProvider)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationCategoryMismatchError,
)
from intergrax.integrations.registry.catalog import get_entry
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderConfigurationError,
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
    EmbeddingProviderRuntimeBindingError,
)
from intergrax.rag.embedding.contracts.runtime_binding_spec import (
    EmbeddingProviderRuntimeBindingSpec,
)
from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig
from intergrax.rag.embedding.registry.provider_authority import (
    default_embedding_provider_slug,
    validate_embedding_provider_slug,
)
from intergrax.rag.embedding.registry.profile import EmbeddingProfile

_EMBEDDING_CATEGORY = IntegrationCategory.EMBEDDING_PROVIDER


def resolve_embedding_provider_slug(
    *,
    integration_profile: IntegrationProfile | None = None,
    provider_id: str | None = None,
    embedding_profile: EmbeddingProfile | None = None,
) -> str:
    """
    Resolve canonical embedding provider slug.

    Precedence:
    explicit IntegrationProfile.embedding_provider
        > explicit provider_id compatibility argument
        > EmbeddingProfile / env compatibility
        > documented default
    """
    from_profile: str | None = None
    if integration_profile is not None:
        from_profile = integration_profile.slug_for_category(_EMBEDDING_CATEGORY)
        if from_profile is not None:
            from_profile = validate_embedding_provider_slug(from_profile)

    from_arg: str | None = None
    if provider_id is not None and provider_id.strip():
        from_arg = validate_embedding_provider_slug(provider_id)

    from_compat: str | None = None
    if embedding_profile is not None:
        from_compat = validate_embedding_provider_slug(embedding_profile.provider)

    explicit_authorities = [value for value in (from_profile, from_arg) if value is not None]
    if len(set(explicit_authorities)) > 1:
        raise EmbeddingProviderConfigurationError(
            "conflicting embedding provider selection: "
            f"integration_profile={from_profile!r}, provider_id={from_arg!r}"
        )

    if from_profile is not None:
        return from_profile

    if from_arg is not None:
        return from_arg

    if from_compat is not None:
        return from_compat

    return validate_embedding_provider_slug(default_embedding_provider_slug())


def _embedding_contract_spec(slug: str):
    entry = get_entry(slug)
    if _EMBEDDING_CATEGORY not in entry.categories:
        raise IntegrationCategoryMismatchError(slug, _EMBEDDING_CATEGORY.value)
    specs = [spec for spec in entry.contract_specs if spec.category == _EMBEDDING_CATEGORY.value]
    if not specs:
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider {slug!r} has no embedding_provider contract spec"
        )
    return specs[0]


def _embedding_runtime_binder(slug: str, spec) -> EmbeddingProviderRuntimeBinder:
    runtime_binding = spec.runtime_binding
    if runtime_binding is None:
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider {slug!r} has no runtime binder registered"
        )
    if not isinstance(runtime_binding, EmbeddingProviderRuntimeBindingSpec):
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider {slug!r} supports runtime binding but runtime binding "
            "descriptor is not compatible with embedding_provider"
        )
    binder = runtime_binding.binder
    if not isinstance(binder, EmbeddingProviderRuntimeBinder):
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider {slug!r} runtime binder is not typed"
        )
    return binder


def bind_embedding_provider(
    *,
    integration_profile: IntegrationProfile | None = None,
    provider_id: str | None = None,
    embedding_profile: EmbeddingProfile | None = None,
    execution_config: EmbeddingProviderExecutionConfig | None = None,
) -> EmbeddingProvider:
    """Bind a typed EmbeddingProvider via Integrations catalog runtime metadata."""
    resolved_slug = resolve_embedding_provider_slug(
        integration_profile=integration_profile,
        provider_id=provider_id,
        embedding_profile=embedding_profile,
    )
    resolved_model = None
    if embedding_profile is not None:
        resolved_model = embedding_profile.model

    profile = integration_profile or IntegrationProfile(embedding_provider=resolved_slug)
    integration_options = profile.options_for_slug(resolved_slug)

    spec = _embedding_contract_spec(resolved_slug)
    if not spec.supports_runtime_binding:
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider {resolved_slug!r} does not support runtime binding"
        )

    binder = _embedding_runtime_binder(resolved_slug, spec)

    context = EmbeddingProviderRuntimeBindingContext(
        provider_slug=resolved_slug,
        model=resolved_model,
        execution_config=execution_config,
        integration_options=integration_options,
    )
    provider = binder.bind(context)
    if provider.provider_name() != resolved_slug:
        raise EmbeddingProviderRuntimeBindingError(
            f"embedding provider runtime binder for {resolved_slug!r} returned "
            f"provider_name={provider.provider_name()!r}"
        )
    return provider


__all__ = [
    "bind_embedding_provider",
    "resolve_embedding_provider_slug",
]
