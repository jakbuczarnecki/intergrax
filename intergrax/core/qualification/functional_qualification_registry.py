# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Explicit plugin registry for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from intergrax.core.qualification.functional_qualification_identity import FunctionalQualificationPluginId
from intergrax.core.qualification.functional_qualification_plugin import FunctionalQualificationPlugin


class QualificationPluginRegistryError(Exception):
    """Registry configuration or lookup failure."""


class QualificationPluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[FunctionalQualificationPluginId, FunctionalQualificationPlugin] = {}
        self._registration_order: list[FunctionalQualificationPluginId] = []

    def register(self, plugin: FunctionalQualificationPlugin) -> None:
        plugin_id = plugin.descriptor.plugin_id
        if plugin_id in self._plugins:
            raise QualificationPluginRegistryError(
                f"duplicate_plugin_id:{plugin_id.value}",
            )
        self._plugins[plugin_id] = plugin
        self._registration_order.append(plugin_id)

    def get(self, plugin_id: FunctionalQualificationPluginId) -> FunctionalQualificationPlugin:
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            raise QualificationPluginRegistryError(f"unknown_plugin_id:{plugin_id.value}")
        return plugin

    def list_plugins(self) -> tuple[FunctionalQualificationPlugin, ...]:
        return tuple(self._plugins[plugin_id] for plugin_id in self._registration_order)

    def list_plugin_ids(self) -> tuple[FunctionalQualificationPluginId, ...]:
        return tuple(self._registration_order)


__all__ = [
    "QualificationPluginRegistry",
    "QualificationPluginRegistryError",
]
