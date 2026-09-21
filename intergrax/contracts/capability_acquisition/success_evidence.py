# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Success evidence invariants for capability realization (UCA-2R)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey


def validate_realization_success_evidence(
    *,
    identity: CapabilityIdentityKey,
    evidence: CapabilityRealizationEvidence | None,
) -> None:
    """Require exact identity HOST_AVAILABLE proof without contradictory disposition keys."""
    if evidence is None or evidence.availability_evidence is None:
        raise ValueError(
            "SUCCEEDED requires availability evidence for canonical projection",
        )
    _validate_availability_proves_host_available(
        identity=identity,
        availability=evidence.availability_evidence,
    )


def validate_availability_proves_host_available(
    *,
    identity: CapabilityIdentityKey,
    availability: CapabilityDiscoveryAvailabilityEvidence,
) -> None:
    """Shared invariant for generic and Tool-domain success outcomes."""
    _validate_availability_proves_host_available(
        identity=identity,
        availability=availability,
    )


def _validate_availability_proves_host_available(
    *,
    identity: CapabilityIdentityKey,
    availability: CapabilityDiscoveryAvailabilityEvidence,
) -> None:
    sort_key = identity.sort_key
    host_keys = frozenset(key.sort_key for key in availability.host_available_keys)
    blocked_keys = frozenset(key.sort_key for key in availability.blocked_keys)
    unavailable_keys = frozenset(key.sort_key for key in availability.unavailable_keys)

    if sort_key not in host_keys:
        raise ValueError(
            "SUCCEEDED requires exact capability_identity in host_available_keys",
        )
    if sort_key in blocked_keys:
        raise ValueError(
            "SUCCEEDED capability_identity must not appear in blocked_keys",
        )
    if sort_key in unavailable_keys:
        raise ValueError(
            "SUCCEEDED capability_identity must not appear in unavailable_keys",
        )
    scope_visible = availability.scope_visible_keys
    if scope_visible is not None:
        visible = frozenset(key.sort_key for key in scope_visible)
        if sort_key not in visible:
            raise ValueError(
                "SUCCEEDED capability_identity must be scope-visible when "
                "scope_visible_keys is present",
            )


__all__ = [
    "validate_availability_proves_host_available",
    "validate_realization_success_evidence",
]
