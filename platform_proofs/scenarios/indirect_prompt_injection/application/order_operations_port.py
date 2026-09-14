"""Application-owned port for order management integrations."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNotesResponse,
    OrderProviderOrder,
    OrderProviderUpdateResponse,
)


class OrderOperationsPort(Protocol):
    """Stable application boundary consumed by scenario tool handlers."""

    def get_order(self, order_id: str) -> OrderProviderOrder: ...

    def get_notes(self, order_id: str) -> OrderProviderNotesResponse: ...

    def update_shipping_address(
        self,
        order_id: str,
        new_shipping_address: str,
    ) -> OrderProviderUpdateResponse: ...
