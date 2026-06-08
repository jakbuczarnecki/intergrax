# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Allowlisted HTTP client integration contract (T-EXPAND T17)."""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class HttpResponse(BaseModel):
    """Normalized HTTP response for agent tools."""

    status_code: int
    body: str = ""
    headers: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class HttpClientBackend(Protocol):
    """Policy-scoped HTTP facade — host allowlist enforced by implementation."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
        timeout_s: float = 30.0,
    ) -> HttpResponse:
        """Execute one HTTP request when URL host is allowlisted."""
