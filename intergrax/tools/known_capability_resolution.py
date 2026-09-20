# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Exact tool package resolution validation for known capability identities (UCA-2R)."""

from __future__ import annotations

from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.tools.catalog import ToolPackageResolution
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError


def assert_exact_tool_package_resolution_for_identity(
    *,
    capability_identity: CapabilityIdentityKey,
    resolution: ToolPackageResolution,
) -> None:
    """Fail closed when resolver output does not match requested capability identity."""
    if capability_identity.kind is not CapabilityKind.TOOL:
        raise DynamicToolAcquisitionResolutionError(
            "capability identity kind must be TOOL for tool package resolution",
        )
    entry = resolution.entry
    candidate = resolution.package_candidate
    if entry.catalog_source_id != capability_identity.source_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolved catalog source id does not match capability identity",
        )
    if entry.logical_tool_id != capability_identity.logical_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolved entry logical tool id does not match capability identity",
        )
    if candidate.logical_tool_id != capability_identity.logical_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolved package logical tool id does not match capability identity",
        )
    if entry.logical_tool_id != candidate.logical_tool_id:
        raise DynamicToolAcquisitionResolutionError(
            "resolution entry and package candidate logical tool ids diverge",
        )
    if entry.package_reference != candidate.package_reference:
        raise DynamicToolAcquisitionResolutionError(
            "resolution entry and package candidate package references diverge",
        )


__all__ = ["assert_exact_tool_package_resolution_for_identity"]
