# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical availability projection from realization evidence (UCA-2)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.scope import CapabilityDiscoveryScopeMode

from intergrax.capability_catalog.discovery import resolve_availability_disposition


def project_availability_disposition(
    *,
    identity: CapabilityIdentityKey,
    evidence: CapabilityRealizationEvidence,
    scope_mode: CapabilityDiscoveryScopeMode = CapabilityDiscoveryScopeMode.GLOBAL,
) -> AvailabilityDisposition:
    """Project HOST_AVAILABLE only when canonical evidence supports it."""
    availability = evidence.availability_evidence
    if availability is None:
        return AvailabilityDisposition.CATALOG_AVAILABLE

    blocked = frozenset(key.sort_key for key in availability.blocked_keys)
    unavailable = frozenset(key.sort_key for key in availability.unavailable_keys)
    host = frozenset(key.sort_key for key in availability.host_available_keys)
    scope_visible = (
        frozenset(key.sort_key for key in availability.scope_visible_keys)
        if availability.scope_visible_keys is not None
        else None
    )
    return resolve_availability_disposition(
        identity_key=identity,
        scope_mode=scope_mode,
        blocked_keys=blocked,
        unavailable_keys=unavailable,
        host_available_keys=host,
        scope_visible_keys=scope_visible,
    )


__all__ = ["project_availability_disposition"]
