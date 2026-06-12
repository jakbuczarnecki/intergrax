# © Artur Czarnecki. All rights reserved.

"""Prometheus SLI bridge (ECP-2.3 / ECP-PROD)."""

from __future__ import annotations

import os

import httpx


def query_gauge(promql: str, *, default: float = 0.0) -> float:
    """Query Prometheus instant vector when INTERGRAX_PROMETHEUS_URL is set."""
    base_url = os.environ.get("INTERGRAX_PROMETHEUS_URL", "").strip()
    if not base_url:
        return default
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=10.0) as client:
            response = client.get("/api/v1/query", params={"query": promql})
            response.raise_for_status()
            body = response.json()
            result = body.get("data", {}).get("result", [])
            if not result:
                return default
            return float(result[0]["value"][1])
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return default
