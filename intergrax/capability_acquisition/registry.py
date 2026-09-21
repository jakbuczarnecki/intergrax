# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin registry for capability realization providers (UCA-2)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.errors import (
    CapabilityRealizationConfigurationError,
)
from intergrax.contracts.capability_acquisition.provider import (
    CapabilityRealizationProvider,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind


class CapabilityRealizationProviderRegistry:
    """Deterministic kind routing — at most one provider per capability kind."""

    def __init__(
        self,
        providers: tuple[CapabilityRealizationProvider, ...],
    ) -> None:
        provider_ids: set[str] = set()
        kind_owners: dict[CapabilityKind, CapabilityRealizationProvider] = {}
        ordered: list[CapabilityRealizationProvider] = []
        for provider in providers:
            provider_id = provider.provider_id
            if provider_id in provider_ids:
                raise CapabilityRealizationConfigurationError(
                    f"duplicate provider_id: {provider_id!r}",
                )
            provider_ids.add(provider_id)
            for kind in provider.supported_kinds:
                if kind in kind_owners:
                    raise CapabilityRealizationConfigurationError(
                        "ambiguous provider for capability kind "
                        f"{kind.value!r}: "
                        f"{kind_owners[kind].provider_id!r} and {provider_id!r}",
                    )
                kind_owners[kind] = provider
            ordered.append(provider)
        self._providers = tuple(ordered)
        self._kind_owners = kind_owners

    @property
    def providers(self) -> tuple[CapabilityRealizationProvider, ...]:
        return self._providers

    def resolve(self, request: CapabilityRealizationRequest) -> CapabilityRealizationProvider | None:
        kind = request.capability_kind
        provider = self._kind_owners.get(kind)
        if provider is None:
            return None
        if not provider.supports(request):
            return None
        return provider

    def eligible_providers(
        self,
        request: CapabilityRealizationRequest,
    ) -> tuple[CapabilityRealizationProvider, ...]:
        """Providers that support the request — used for ambiguity detection."""
        return tuple(
            provider
            for provider in self._providers
            if provider.supports(request)
        )


__all__ = ["CapabilityRealizationProviderRegistry"]
