# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inbound interaction contract (integration catalog surface)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.integrations.contracts.inbound_interaction import InboundInteraction


@runtime_checkable
class InteractionSurface(Protocol):
    @property
    def channel(self) -> str: ...

    def can_handle(self, payload: dict[str, Any]) -> bool: ...

    def to_inbound(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str,
        user_id: str,
    ) -> InboundInteraction: ...


InteractionAdapter = InteractionSurface

__all__ = ["InteractionAdapter", "InteractionSurface", "InboundInteraction"]
