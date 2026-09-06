# © Artur Czarnecki. All rights reserved.

"""Default capability health provider composition (P1.5)."""

from __future__ import annotations

from intergrax.applications._shared.capability_health.providers.dependency_validation import (
    dependency_validation_health_provider,
)
from intergrax.applications._shared.capability_health.providers.tool_availability import (
    tool_effective_availability_health_provider,
)
from intergrax.applications.contracts.capability_health import CapabilityHealthProvider


def default_capability_health_providers() -> tuple[CapabilityHealthProvider, ...]:
    """Explicit immutable provider set — no global mutable registry."""
    return (
        dependency_validation_health_provider(),
        tool_effective_availability_health_provider(),
    )
