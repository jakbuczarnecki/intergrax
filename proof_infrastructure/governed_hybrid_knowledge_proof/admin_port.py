# © Artur Czarnecki. All rights reserved.

"""Proof-only administration port for the Dockerized security-status vendor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from proof_infrastructure.controlled_security_status_service.models import (
    SecurityStatusReadBehaviorControlV1,
    SecurityStatusReadBehaviorV1,
    SecurityStatusRefreshControlV1,
    SecurityStatusResponseV1,
)


class SecurityStatusFixtureIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(..., min_length=1, max_length=64)
    status: str = Field(..., min_length=1, max_length=32)
    updated_at: datetime


@runtime_checkable
class ControlledSecurityStatusAdminPort(Protocol):
    def seed_security_status(self) -> SecurityStatusFixtureIdentityV1:
        """Seed the canonical ORION security fixture through vendor control plane."""

    def set_read_behavior(self, behavior: SecurityStatusReadBehaviorV1) -> None:
        """Inject vendor read behavior for failure/recovery scenarios."""

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        """Wait until the vendor reports healthy readiness."""

    def read_request_count(self) -> int:
        """Return vendor-side live read request count."""

    def reset_read_request_count(self) -> None:
        """Reset vendor-side live read request counter."""

    def read_safe_fixture_identity(self, *, project_id: str) -> SecurityStatusFixtureIdentityV1:
        """Read fixture identity without incrementing live read counters."""

    def refresh_security_status(self, *, updated_at: datetime) -> SecurityStatusFixtureIdentityV1:
        """Persist refreshed security evidence timestamp through vendor control plane."""


def _fixture_identity_from_response(response: SecurityStatusResponseV1) -> SecurityStatusFixtureIdentityV1:
    return SecurityStatusFixtureIdentityV1(
        project_id=response.project_id,
        status=response.status,
        updated_at=response.updated_at,
    )


@dataclass(frozen=True, slots=True)
class HttpxControlledSecurityStatusAdminPort:
    """HTTP adapter for proof control operations against the controlled vendor."""

    base_url: str
    timeout_seconds: float = 5.0

    def seed_security_status(self) -> SecurityStatusFixtureIdentityV1:
        response = httpx.post(
            f"{self.base_url}/control/seed-orion",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = SecurityStatusResponseV1.model_validate(response.json())
        return _fixture_identity_from_response(payload)

    def set_read_behavior(self, behavior: SecurityStatusReadBehaviorV1) -> None:
        control = SecurityStatusReadBehaviorControlV1(behavior=behavior)
        response = httpx.put(
            f"{self.base_url}/control/read-behavior",
            json=control.model_dump(mode="json"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def wait_until_ready(self, *, timeout_seconds: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "startup_timeout"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        raise RuntimeError(f"security_vendor_unavailable: {last_error}")

    def read_request_count(self) -> int:
        response = httpx.get(
            f"{self.base_url}/control/request-count",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return int(response.json()["read_request_count"])

    def reset_read_request_count(self) -> None:
        response = httpx.post(
            f"{self.base_url}/control/request-count/reset",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def read_safe_fixture_identity(self, *, project_id: str) -> SecurityStatusFixtureIdentityV1:
        response = httpx.get(
            f"{self.base_url}/control/fixture/{project_id}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = SecurityStatusResponseV1.model_validate(response.json())
        return _fixture_identity_from_response(payload)

    def refresh_security_status(self, *, updated_at: datetime) -> SecurityStatusFixtureIdentityV1:
        control = SecurityStatusRefreshControlV1(updated_at=updated_at)
        response = httpx.post(
            f"{self.base_url}/control/refresh-orion",
            json=control.model_dump(mode="json"),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = SecurityStatusResponseV1.model_validate(response.json())
        return _fixture_identity_from_response(payload)
