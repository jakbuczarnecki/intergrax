"""In-process order provider for unit and integration tests."""

from __future__ import annotations

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderMutation,
    OrderProviderNote,
    OrderProviderNotesResponse,
    OrderProviderOrder,
    OrderProviderUpdateResponse,
    ProviderMutationState,
)
from platform_proofs.scenarios.indirect_prompt_injection.provider.order_service.app import (
    DEFAULT_ORDER_ID,
    DEFAULT_SHIPPING_ADDRESS,
    _Store,
)


class InProcessOrderProviderClient(OrderProviderClient):
    def __init__(self) -> None:
        self._store = _Store()

    def reset(self, *, notes: list[OrderProviderNote] | None = None) -> None:
        serialized = [
            note.model_dump(mode="json")
            if isinstance(note, OrderProviderNote)
            else OrderProviderNote.model_validate(note).model_dump(mode="json")
            for note in (notes or [])
        ]
        self._store.reset(notes=serialized)

    def get_order(self, order_id: str) -> OrderProviderOrder:
        order = self._store.get_order(order_id)
        return OrderProviderOrder(
            order_id=order.order_id,
            status=order.status,
            shipping_address=order.shipping_address,
            fulfillment_status=order.fulfillment_status,
        )

    def get_notes(self, order_id: str) -> OrderProviderNotesResponse:
        return OrderProviderNotesResponse(
            order_id=order_id,
            notes=[
                OrderProviderNote.model_validate(note)
                for note in self._store.get_notes(order_id)
            ],
        )

    def update_shipping_address(
        self,
        order_id: str,
        new_shipping_address: str,
    ) -> OrderProviderUpdateResponse:
        order = self._store.update_shipping_address(order_id, new_shipping_address)
        return OrderProviderUpdateResponse(
            order_id=order.order_id,
            status=order.status,
            shipping_address=order.shipping_address,
            fulfillment_status=order.fulfillment_status,
            confirmation="shipping_address_updated",
        )

    def mutation_state(self) -> ProviderMutationState:
        mutations = self._store.mutations()
        return ProviderMutationState(
            write_count=len(mutations),
            mutations=tuple(
                OrderProviderMutation(
                    method=item.method,
                    path=item.path,
                    payload=dict(item.payload),
                )
                for item in mutations
            ),
        )


DEFAULT_TEST_ORDER_ID = DEFAULT_ORDER_ID
DEFAULT_TEST_ADDRESS = DEFAULT_SHIPPING_ADDRESS
