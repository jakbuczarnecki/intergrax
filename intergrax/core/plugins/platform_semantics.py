# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Platform Plugin lifecycle and conflict vocabulary (PLATFORM-PLUGIN-6)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.core.distribution import (
    PlatformCompatibilityResult,
    check_platform_compatibility,
)
from intergrax.core.plugins.package_contract import PlatformPluginManifest


class PlatformPluginLifecycleState(StrEnum):
    """Cross-cutting lifecycle vocabulary (observability and coordination only)."""

    DISCOVERED = "discovered"
    VALIDATED = "validated"
    ENABLED = "enabled"
    MATERIALIZED = "materialized"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class PlatformPluginConflictKind(StrEnum):
    """Shared conflict classification; resolution policy remains domain-owned."""

    PACKAGE_IDENTITY = "package_identity"
    ENTRY_POINT_NAME = "entry_point_name"
    CAPABILITY_IDENTITY = "capability_identity"
    DOMAIN_RESOURCE_ID = "domain_resource_id"


def check_manifest_platform_compatibility(
    manifest: PlatformPluginManifest,
    platform_version: str,
) -> PlatformCompatibilityResult:
    """Check package-level platform compatibility for a manifest."""
    return check_platform_compatibility(manifest.platform_compatibility, platform_version)
