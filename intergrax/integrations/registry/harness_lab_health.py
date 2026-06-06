# © Artur Czarnecki. All rights reserved.

"""Harness lab integration health probes with circuit breaker (Phase W-OPS.10)."""

from __future__ import annotations

from intergrax.integrations._shared.health import health_check_catalog_slugs
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.registry.harness_lab_stack import (
    HARNESS_LAB_STABLE_SLUGS,
    HARNESS_M6_P4_PROBE_SLUGS,
    HARNESS_M6_P5_PROBE_SLUGS,
)


def health_check_harness_lab_stack() -> list[HealthStatus]:
    """
    Probe every harness lab stable catalog slug using circuit-breaker protected resolution.

    Requires shipped integrations to be bootstrapped (``register_default_integrations``).
    """
    return health_check_catalog_slugs(
        sorted(HARNESS_LAB_STABLE_SLUGS),
        use_circuit_breaker=True,
    )


def health_check_harness_m6_p4_probes() -> list[HealthStatus]:
    """Probe M.6 P4 harness-ROI slugs (W-OPS.10 extension)."""
    return health_check_catalog_slugs(
        sorted(HARNESS_M6_P4_PROBE_SLUGS),
        use_circuit_breaker=True,
    )


def health_check_harness_m6_p5_probes() -> list[HealthStatus]:
    """Probe M.6 P5 harness-depth slugs (W-OPS.10 extension)."""
    return health_check_catalog_slugs(
        sorted(HARNESS_M6_P5_PROBE_SLUGS),
        use_circuit_breaker=True,
    )
