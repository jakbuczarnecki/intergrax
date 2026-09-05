# © Artur Czarnecki. All rights reserved.

"""Deterministic effective profile fingerprint (P1.1)."""

from __future__ import annotations

from intergrax.applications._shared.environment_snapshot_wiring import stable_digest_hex
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    bundle_normalized_payload,
)


def compute_effective_profile_fingerprint(profile: ApplicationEnvironmentProfile) -> str:
    """Return deterministic semantic fingerprint for an effective profile."""
    payload = bundle_normalized_payload(profile.bundle_dump(mode="json"))
    return stable_digest_hex(payload)
