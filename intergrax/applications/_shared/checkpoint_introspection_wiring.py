# © Artur Czarnecki. All rights reserved.

"""Checkpoint introspection API wiring for product hosts (AUDIT-IDEAL-8.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class CheckpointIntrospectionWiring:
    enabled: bool
    route_prefix: str


def resolve_checkpoint_introspection_wiring(
    env: ApplicationEnvironmentProfile,
) -> CheckpointIntrospectionWiring:
    """Product hosts expose read-only checkpoint introspection for ops."""
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    enabled = is_product and env.features.checkpoint_introspection_enabled
    return CheckpointIntrospectionWiring(enabled=enabled, route_prefix="/v1/tasks")
