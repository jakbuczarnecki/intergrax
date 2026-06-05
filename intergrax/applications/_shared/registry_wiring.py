# © Artur Czarnecki. All rights reserved.

"""Tier-3 harness registry wiring (Phase REG-1)."""

from __future__ import annotations

from intergrax.applications._shared.registry_snapshot import (
    HarnessRegistrySnapshot,
    resolve_registry_snapshot,
)
from intergrax.applications._shared.registry_snapshot_protocol import RegistrySnapshotProtocol
from intergrax.applications.contracts.build_context import ApplicationBuildContext


def resolve_registry_snapshot_protocol(
    ctx: ApplicationBuildContext,
) -> RegistrySnapshotProtocol:
    """Return snapshot as :class:`RegistrySnapshotProtocol` for conformance checks."""
    snapshot = resolve_registry_snapshot(ctx)
    return snapshot


__all__ = [
    "HarnessRegistrySnapshot",
    "resolve_registry_snapshot",
    "resolve_registry_snapshot_protocol",
]
