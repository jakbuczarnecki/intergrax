"""HTTP client for the controlled external Order Service."""

from __future__ import annotations

import os

import httpx

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderMutation,
    OrderProviderNote,
    OrderProviderNotesResponse,
    OrderProviderOrder,
    OrderProviderResetRequest,
    OrderProviderUpdateResponse,
    ProviderMutationState,
)

DEFAULT_ORDER_SERVICE_BASE_URL = "http://127.0.0.1:18091"


class OrderProviderClient:
    def __init__(self, base_url: str | None = None, *, timeout_seconds: float = 10.0) -> None:
        resolved = base_url or os.environ.get("INTERGRAX_ORDER_SERVICE_URL", DEFAULT_ORDER_SERVICE_BASE_URL)
        self._base_url = resolved.rstrip("/")
        self._timeout = timeout_seconds

    def reset(self, *, notes: list[OrderProviderNote] | None = None) -> None:
        body = OrderProviderResetRequest(notes=list(notes or []))
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/debug/reset",
                json=body.model_dump(mode="json"),
            )
            response.raise_for_status()

    def get_order(self, order_id: str) -> OrderProviderOrder:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/orders/{order_id}")
            response.raise_for_status()
            return OrderProviderOrder.model_validate(response.json())

    def get_notes(self, order_id: str) -> OrderProviderNotesResponse:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/orders/{order_id}/notes")
            response.raise_for_status()
            return OrderProviderNotesResponse.model_validate(response.json())

    def update_shipping_address(
        self,
        order_id: str,
        new_shipping_address: str,
    ) -> OrderProviderUpdateResponse:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.patch(
                f"{self._base_url}/orders/{order_id}/shipping-address",
                json={"new_shipping_address": new_shipping_address},
            )
            response.raise_for_status()
            return OrderProviderUpdateResponse.model_validate(response.json())

    def mutation_state(self) -> ProviderMutationState:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/debug/mutations")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("order_provider_invalid_response")
            mutations_raw = payload.get("mutations", [])
            mutations = tuple(
                OrderProviderMutation.model_validate(item)
                for item in mutations_raw
                if isinstance(item, dict)
            )
            write_count = int(payload.get("write_count", len(mutations)))
            return ProviderMutationState(write_count=write_count, mutations=mutations)
