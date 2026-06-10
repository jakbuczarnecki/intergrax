# © Artur Czarnecki. All rights reserved.

"""Product long-running resume wiring (AUDIT-IDEAL-8.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class ProductLongRunningWiring:
    scheduler_enabled: bool
    checkpoint_resume_enabled: bool


def resolve_product_long_running_wiring(
    env: ApplicationEnvironmentProfile,
) -> ProductLongRunningWiring:
    """Product hosts enable scheduler + checkpoint resume when reliability profile allows."""
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    scheduler = env.reliability_profile.long_running_scheduler_enabled
    features = env.features.long_running_scheduler
    return ProductLongRunningWiring(
        scheduler_enabled=is_product and (scheduler or features),
        checkpoint_resume_enabled=is_product and scheduler,
    )
