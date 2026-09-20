# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Capability acquisition strategy SPI (UCA-3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.kind import CapabilityKind


@runtime_checkable
class CapabilityAcquisitionStrategy(Protocol):
    """Replaceable acquisition adapter — not a domain lifecycle or execution owner."""

    @property
    def strategy_id(self) -> str:
        """Stable strategy identity."""
        ...

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        """Capability kinds this strategy declares it can acquire (integrity routing)."""
        ...

    def supports(self, request: CapabilityAcquisitionRequest) -> bool:
        """Whether this strategy can technically handle the request."""
        ...

    def acquire(
        self, request: CapabilityAcquisitionRequest
    ) -> CapabilityAcquisitionResult:
        """Perform strategy-owned acquisition handoff for one request.

        Idempotency: for the same ``request_id`` and immutable request semantics,
        the strategy MUST NOT duplicate unsafe domain side effects (delegate to
        the domain lifecycle owner when needed).

        Strategies MUST NOT self-authorize, execute workloads, or mint authority.
        """
        ...


__all__ = ["CapabilityAcquisitionStrategy"]
