# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Plugin registry for capability qualification providers (UCA-4)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.errors import (
    CapabilityQualificationConfigurationError,
)
from intergrax.contracts.capability_qualification.provider import (
    CapabilityQualificationProvider,
)
from intergrax.contracts.capability_qualification.provider_descriptor import (
    CapabilityQualificationProviderDescriptor,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)


def descriptor_for_provider(
    provider: CapabilityQualificationProvider,
) -> CapabilityQualificationProviderDescriptor:
    return CapabilityQualificationProviderDescriptor(provider_id=provider.provider_id)


class CapabilityQualificationProviderRegistry:
    """Stores providers and resolves eligibility — not a policy engine."""

    def __init__(
        self,
        providers: tuple[CapabilityQualificationProvider, ...],
    ) -> None:
        provider_ids: set[str] = set()
        ordered: list[CapabilityQualificationProvider] = []
        for provider in providers:
            provider_id = provider.provider_id
            if provider_id in provider_ids:
                raise CapabilityQualificationConfigurationError(
                    f"duplicate provider_id: {provider_id!r}",
                )
            provider_ids.add(provider_id)
            ordered.append(provider)
        self._providers = tuple(ordered)
        self._by_id = {provider.provider_id: provider for provider in self._providers}

    @property
    def providers(self) -> tuple[CapabilityQualificationProvider, ...]:
        return self._providers

    def get(self, provider_id: str) -> CapabilityQualificationProvider | None:
        return self._by_id.get(provider_id)

    def eligible_providers(
        self,
        request: CapabilityQualificationRequest,
    ) -> tuple[CapabilityQualificationProvider, ...]:
        eligible: list[CapabilityQualificationProvider] = []
        for provider in self._providers:
            if provider.supports(request):
                eligible.append(provider)
        return tuple(eligible)

    def eligible_descriptors(
        self,
        request: CapabilityQualificationRequest,
    ) -> tuple[CapabilityQualificationProviderDescriptor, ...]:
        eligible = self.eligible_providers(request)
        return tuple(descriptor_for_provider(provider) for provider in eligible)

    def descriptors_for_eligible(
        self,
        eligible: tuple[CapabilityQualificationProvider, ...],
    ) -> tuple[CapabilityQualificationProviderDescriptor, ...]:
        return tuple(descriptor_for_provider(provider) for provider in eligible)


__all__ = [
    "CapabilityQualificationProviderRegistry",
    "descriptor_for_provider",
]
