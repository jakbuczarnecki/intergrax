# © Artur Czarnecki. All rights reserved.

"""Canary traffic routing for adaptive profile versions (Phase W-ADAPT-4.3)."""

from __future__ import annotations

import hashlib


def should_route_canary_traffic(
    *,
    tenant_id: str,
    routing_key: str,
    canary_tenant_allowlist: list[str],
    canary_traffic_percent: float,
) -> bool:
    """
    Decide whether a request should use the candidate/canary profile version.

    Allowlisted tenants always receive canary traffic. Otherwise a deterministic
    hash bucket selects ``canary_traffic_percent`` of traffic.
    """
    if tenant_id in canary_tenant_allowlist:
        return True
    if canary_traffic_percent <= 0.0:
        return False
    if canary_traffic_percent >= 100.0:
        return True
    digest = hashlib.sha256(f"{tenant_id}:{routing_key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    threshold = int(canary_traffic_percent * 100.0)
    return bucket < threshold
