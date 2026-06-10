# © Artur Czarnecki. All rights reserved.

"""Production Celery/K8s capacity adapter wiring (AUDIT-IDEAL-30.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.capacity.production_adapters import (
    ProductionCapacityAdapters,
    apply_production_scale_probe,
    build_production_capacity_adapters,
)


@dataclass(frozen=True, slots=True)
class ProductionCapacityWiring:
    enabled: bool
    adapters: ProductionCapacityAdapters | None
    probe_passed: bool


def resolve_production_capacity_wiring(
    env: ApplicationEnvironmentProfile,
) -> ProductionCapacityWiring:
    """Enable production-scale Celery/K8s adapters on product hosts."""
    scaling = env.scaling_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ProductionCapacityWiring(enabled=False, adapters=None, probe_passed=False)
    if not scaling.production_adapters_enabled:
        return ProductionCapacityWiring(enabled=False, adapters=None, probe_passed=False)

    adapters = build_production_capacity_adapters()
    probe_passed = apply_production_scale_probe(adapters)
    return ProductionCapacityWiring(enabled=True, adapters=adapters, probe_passed=probe_passed)
