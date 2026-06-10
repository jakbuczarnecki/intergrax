# © Artur Czarnecki. All rights reserved.

"""Product intake wiring helpers (AUDIT-IDEAL-3.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.async_task_index_resolver import resolve_async_task_index
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class ProductIntakeWiring:
    """Resolved product intake surfaces."""

    durable_async_index: bool
    streaming_intake_enabled: bool


def resolve_product_intake_wiring(env: ApplicationEnvironmentProfile) -> ProductIntakeWiring:
    """Return durable async + streaming intake flags for product hosts."""
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    durable = is_product and env.features.durable_async_index_default
    streaming = is_product and env.features.streaming_intake_enabled
    if durable:
        _ = resolve_async_task_index(env)
    return ProductIntakeWiring(
        durable_async_index=durable,
        streaming_intake_enabled=streaming,
    )
