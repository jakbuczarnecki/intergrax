# © Artur Czarnecki. All rights reserved.

"""Prometheus SLI bridge (ECP-2.3)."""

from __future__ import annotations


def query_gauge(promql: str, *, default: float = 0.0) -> float:
    """Mock PromQL bridge for tests and optional profiles."""
    _ = promql
    return default
