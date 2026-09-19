# © Artur Czarnecki. All rights reserved.

"""Tier-3 harness registry wiring (Phase REG-1)."""

from __future__ import annotations

from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
)
from intergrax.applications._shared.registry_snapshot import (
    HarnessRegistrySnapshot,
    resolve_registry_snapshot,
)
from intergrax.applications._shared.registry_snapshot_protocol import RegistrySnapshotProtocol


def resolve_registry_snapshot_protocol(
    composition: ApplicationCompositionContext,
) -> RegistrySnapshotProtocol:
    """Return snapshot as :class:`RegistrySnapshotProtocol` for conformance checks."""
    snapshot = resolve_registry_snapshot(composition)
    return snapshot


__all__ = [
    "HarnessRegistrySnapshot",
    "resolve_registry_snapshot",
    "resolve_registry_snapshot_protocol",
]
