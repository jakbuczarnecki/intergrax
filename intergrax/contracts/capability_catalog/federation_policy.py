# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated catalog read resilience policy (ME-11)."""

from __future__ import annotations

from enum import StrEnum


class CapabilityCatalogFederationPolicy(StrEnum):
    """How federation reacts when a catalog source fails during read."""

    STRICT_COMPLETE = "strict_complete"
    """Abort federation — no partial snapshot (default, fail-closed)."""

    ALLOW_PARTIAL = "allow_partial"
    """Return a partial snapshot with explicit unavailable source evidence."""
