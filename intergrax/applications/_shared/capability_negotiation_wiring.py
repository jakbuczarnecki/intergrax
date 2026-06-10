# © Artur Czarnecki. All rights reserved.

"""Capability negotiation at runtime resolve (AUDIT-IDEAL-19.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class CapabilityNegotiationResult:
    requested: tuple[str, ...]
    granted: tuple[str, ...]
    denied: tuple[str, ...]
    negotiated: bool


def negotiate_runtime_capabilities(
    requested_capabilities: tuple[str, ...],
    *,
    available_capabilities: tuple[str, ...],
    env: ApplicationEnvironmentProfile,
) -> CapabilityNegotiationResult:
    """Intersect requested capabilities with the host-available capability set."""
    available = set(available_capabilities)
    granted = tuple(item for item in requested_capabilities if item in available)
    denied = tuple(item for item in requested_capabilities if item not in available)
    if env.application_profile is ApplicationProfile.LAB:
        negotiated = True
    else:
        negotiated = not denied
    return CapabilityNegotiationResult(
        requested=requested_capabilities,
        granted=granted,
        denied=denied,
        negotiated=negotiated,
    )
