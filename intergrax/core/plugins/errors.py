# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Errors for Tier-0 catalog plugin loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.core.plugins.platform_qualification import PluginQualificationResult
    from intergrax.core.plugins.platform_semantics import PlatformPluginConflictKind


class PluginError(Exception):
    """Base error for catalog plugin loading."""


class PluginLoadError(PluginError):
    """Entry point or plugin class failed to load."""


class PluginConflictError(PluginError):
    """Duplicate catalog identity when registering plugins."""

    def __init__(
        self,
        message: str,
        *,
        plugin_name: str = "",
        group: str = "",
        conflict_kind: PlatformPluginConflictKind | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name
        self.group = group
        self.conflict_kind = conflict_kind


class PlatformPluginContractError(PluginError):
    """Base error for package-level Platform Plugin contract handling."""


class PlatformPluginManifestValidationError(PlatformPluginContractError):
    """Platform Plugin manifest or package metadata failed validation."""


class ProductionQualificationRequiredError(PlatformPluginContractError):
    """Production host profile requires production-qualified evidence."""

    def __init__(self, message: str, *, result: PluginQualificationResult | None = None) -> None:
        super().__init__(message)
        self.result = result
