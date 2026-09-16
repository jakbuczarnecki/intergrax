# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit capability-kind → handoff handler mapping (ME-RB4)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.marketplace.lifecycle_handoff_handler import (
    MarketplaceLifecycleHandoffHandler,
)


class LifecycleHandoffResolver:
    """Typed, composition-root supplied handler map — no global registry."""

    def __init__(
        self,
        handlers: Mapping[CapabilityKind, MarketplaceLifecycleHandoffHandler],
    ) -> None:
        self._handlers = dict(handlers)

    def handler_for(self, kind: CapabilityKind) -> MarketplaceLifecycleHandoffHandler | None:
        return self._handlers.get(kind)

    @property
    def registered_kinds(self) -> tuple[CapabilityKind, ...]:
        return tuple(sorted(self._handlers, key=lambda item: item.value))


__all__ = ["LifecycleHandoffResolver"]
