# © Artur Czarnecki. All rights reserved.

"""External oracle client for side-effect proof."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    business_operation_id: str
    effect_count: int
    attempt_count: int


class EffectOracle:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> EffectOracle:
        return cls(os.environ.get("EFFECT_SERVICE_URL", "http://external-effect-service:8080"))

    def wait_healthy(self, timeout_s: float = 120.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=3.0) as client:
                    response = client.get(f"{self._base_url}/health")
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("external effect service not healthy")

    def reset(self) -> None:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{self._base_url}/admin/reset")
            response.raise_for_status()

    def snapshot(self, business_operation_id: str) -> OperationSnapshot:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self._base_url}/admin/effects/{business_operation_id}")
            response.raise_for_status()
            payload = response.json()
        return OperationSnapshot(
            business_operation_id=business_operation_id,
            effect_count=len(payload.get("effects", [])),
            attempt_count=int(payload.get("attempt_count", 0)),
        )

    def wait_for_effect_count(
        self,
        business_operation_id: str,
        *,
        expected: int,
        timeout_s: float = 60.0,
    ) -> OperationSnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.snapshot(business_operation_id)
            if snap.effect_count == expected:
                return snap
            time.sleep(0.2)
        snap = self.snapshot(business_operation_id)
        raise RuntimeError(
            f"expected {expected} effects for {business_operation_id}, got {snap.effect_count}",
        )

    def duplicate_scan(self) -> list[dict[str, Any]]:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self._base_url}/admin/duplicates")
            response.raise_for_status()
            return list(response.json().get("duplicates", []))

    def release_before(self, business_operation_id: str) -> None:
        with httpx.Client(timeout=30.0) as client:
            client.post(f"{self._base_url}/admin/release-before/{business_operation_id}")

    def release_after(self, business_operation_id: str) -> None:
        with httpx.Client(timeout=30.0) as client:
            client.post(f"{self._base_url}/admin/release-after/{business_operation_id}")
