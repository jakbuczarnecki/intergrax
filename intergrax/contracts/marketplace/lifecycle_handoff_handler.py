# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Marketplace lifecycle handoff handler SPI (ME-RB4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.lifecycle_handoff_outcome import (
    MarketplaceLifecycleHandoffOutcome,
)
from intergrax.contracts.marketplace.lifecycle_handoff_request import (
    MarketplaceLifecycleHandoffRequest,
)


@runtime_checkable
class MarketplaceLifecycleHandoffHandler(Protocol):
    """Plugin replacement point — external handlers satisfy this contract."""

    @property
    def capability_kind(self) -> CapabilityKind: ...

    @property
    def domain_authority_id(self) -> str: ...

    def handoff(
        self,
        request: MarketplaceLifecycleHandoffRequest,
    ) -> MarketplaceLifecycleHandoffOutcome: ...


__all__ = ["MarketplaceLifecycleHandoffHandler"]
