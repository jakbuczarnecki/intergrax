# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge handoff envelope facts into usage attribution — no usage emission."""

from __future__ import annotations

from intergrax.capability_metering.attribution import CapabilityUsageAttribution
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffEnvelope


def attribution_from_handoff_envelope(
    envelope: CapabilityHandoffEnvelope,
) -> CapabilityUsageAttribution:
    """Project exact release identity into metering attribution without catalog re-lookup."""
    release = envelope.selected_release
    return CapabilityUsageAttribution(
        identity=CapabilityIdentityKey.from_discovery_identity(release.discovery),
        provenance=release.to_provenance(),
    )


__all__ = ["attribution_from_handoff_envelope"]
