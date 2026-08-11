# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Errors for Tier-0 catalog plugin loading."""

from __future__ import annotations


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
        conflict_kind: object | None = None,
    ) -> None:
        super().__init__(message)
        self.plugin_name = plugin_name
        self.group = group
        self.conflict_kind = conflict_kind


class PlatformPluginContractError(PluginError):
    """Base error for package-level Platform Plugin contract handling."""


class PlatformPluginManifestValidationError(PlatformPluginContractError):
    """Platform Plugin manifest or package metadata failed validation."""


class InvalidPlatformVersionError(PlatformPluginContractError):
    """Platform version string is not a valid PEP 440 version."""


class PlatformIncompatibilityError(PlatformPluginContractError):
    """Declared platform compatibility range does not include the tested version."""

    def __init__(self, message: str, *, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result
