# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin registry for qualified capability binding providers (UCA-6C)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.errors import (
    CapabilityQualificationConfigurationError,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingProvider,
    QualifiedCapabilityBindingRequest,
)


class QualifiedCapabilityBindingProviderRegistry:
    """Deterministic provider selection — ambiguous support is a configuration error."""

    def __init__(
        self,
        providers: tuple[QualifiedCapabilityBindingProvider, ...],
    ) -> None:
        provider_ids: set[str] = set()
        ordered: list[QualifiedCapabilityBindingProvider] = []
        for provider in providers:
            provider_id = provider.provider_id
            if provider_id in provider_ids:
                raise CapabilityQualificationConfigurationError(
                    f"duplicate binding provider_id: {provider_id!r}",
                )
            provider_ids.add(provider_id)
            ordered.append(provider)
        self._providers = tuple(ordered)

    @property
    def providers(self) -> tuple[QualifiedCapabilityBindingProvider, ...]:
        return self._providers

    def eligible_providers(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> tuple[QualifiedCapabilityBindingProvider, ...]:
        return tuple(
            provider for provider in self._providers if provider.supports(request)
        )

    def resolve(
        self,
        request: QualifiedCapabilityBindingRequest,
    ) -> QualifiedCapabilityBindingProvider | None:
        eligible = self.eligible_providers(request)
        if not eligible:
            return None
        if len(eligible) != 1:
            if len(eligible) > 1:
                raise CapabilityQualificationConfigurationError(
                    "ambiguous qualified capability binding providers for request "
                    f"{request.binding_operation_id!r}",
                )
            return None
        return eligible[0]


__all__ = ["QualifiedCapabilityBindingProviderRegistry"]
