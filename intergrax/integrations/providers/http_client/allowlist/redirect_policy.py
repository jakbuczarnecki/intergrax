# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redirect authority, credential isolation, and operation deadline helpers."""

from __future__ import annotations

import time
from urllib.parse import urlparse

import httpx

from intergrax.runtime.execution.deadline_provider_guard import (
    resolve_active_provider_timeout_seconds,
)

_CREDENTIAL_HEADER_NAMES = frozenset({"authorization", "cookie"})


def resolve_http_operation_timeout_seconds(configured_timeout_s: float) -> float:
    """Single wall-clock budget for an HTTP operation including all redirect hops."""
    effective = resolve_active_provider_timeout_seconds(configured_timeout_s)
    if effective is None:
        return configured_timeout_s
    return max(0.0, float(effective))


def operation_deadline_monotonic(configured_timeout_s: float) -> float:
    budget = resolve_http_operation_timeout_seconds(configured_timeout_s)
    return time.monotonic() + budget


def remaining_hop_timeout_s(operation_deadline: float) -> float:
    remaining = operation_deadline - time.monotonic()
    if remaining <= 0:
        raise httpx.TimeoutException("http_operation_deadline_exceeded")
    return remaining


def authority_tuple(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, port


def strip_cross_authority_credentials(
    headers: dict[str, str],
    *,
    from_url: str,
    to_url: str,
) -> dict[str, str]:
    if authority_tuple(from_url) == authority_tuple(to_url):
        return headers
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _CREDENTIAL_HEADER_NAMES
    }


__all__ = [
    "authority_tuple",
    "operation_deadline_monotonic",
    "remaining_hop_timeout_s",
    "resolve_http_operation_timeout_seconds",
    "strip_cross_authority_credentials",
]
