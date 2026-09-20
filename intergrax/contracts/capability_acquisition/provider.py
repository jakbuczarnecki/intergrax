# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability realization provider SPI (UCA-2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind


@runtime_checkable
class CapabilityRealizationProvider(Protocol):
    """Domain-owned realization SPI — replaceable by external implementations."""

    @property
    def provider_id(self) -> str:
        """Stable provider identity."""
        ...

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        """Capability kinds handled by this provider (routing, not ranking)."""
        ...

    def supports(self, request: CapabilityRealizationRequest) -> bool:
        """Whether this provider accepts the request (identity + kind)."""
        ...

    def realize(self, request: CapabilityRealizationRequest) -> CapabilityRealizationResult:
        """Perform domain-owned realization handoff for one request."""
        ...


__all__ = ["CapabilityRealizationProvider"]
