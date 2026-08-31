"""HTTP client for the controlled external Order Service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_ORDER_SERVICE_BASE_URL = "http://127.0.0.1:18091"


@dataclass(frozen=True, slots=True)
class ProviderMutationState:
    write_count: int
    mutations: tuple[dict[str, Any], ...]


class OrderProviderClient:
    def __init__(self, base_url: str | None = None, *, timeout_seconds: float = 10.0) -> None:
        resolved = base_url or os.environ.get("INTERGRAX_ORDER_SERVICE_URL", DEFAULT_ORDER_SERVICE_BASE_URL)
        self._base_url = resolved.rstrip("/")
        self._timeout = timeout_seconds

    def reset(self, *, notes: list[dict[str, Any]] | None = None) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/debug/reset",
                json={"notes": notes or []},
            )
            response.raise_for_status()

    def get_order(self, order_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/orders/{order_id}")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("order_provider_invalid_response")
            return payload

    def get_notes(self, order_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/orders/{order_id}/notes")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("order_provider_invalid_response")
            return payload

    def update_shipping_address(self, order_id: str, new_shipping_address: str) -> dict[str, Any]:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.patch(
                f"{self._base_url}/orders/{order_id}/shipping-address",
                json={"new_shipping_address": new_shipping_address},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("order_provider_invalid_response")
            return payload

    def mutation_state(self) -> ProviderMutationState:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(f"{self._base_url}/debug/mutations")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError("order_provider_invalid_response")
            mutations_raw = payload.get("mutations", [])
            mutations = tuple(item for item in mutations_raw if isinstance(item, dict))
            write_count = int(payload.get("write_count", len(mutations)))
            return ProviderMutationState(write_count=write_count, mutations=mutations)
