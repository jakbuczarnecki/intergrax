# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin discovery (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.core.plugins.discovery import (
    EP_MEMORY_STORES,
    EntryPointLoadResult,
    LoadIsolation,
    load_entry_point_targets,
)
from intergrax.core.plugins.errors import PluginLoadError
from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    classify_memory_store_plugin_record,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError


def _plugin_type_from_entry_point_target(target: object, value: str) -> type:
    if isinstance(target, type):
        return target
    if callable(target):
        try:
            loaded = target()
        except Exception as exc:
            raise PluginLoadError(f"Failed to call entry point factory {value!r}: {exc}") from exc
        if not isinstance(loaded, type):
            raise PluginLoadError(f"Factory {value!r} must return a plugin class")
        return loaded
    raise PluginLoadError(f"Entry point {value!r} is not a class or factory")


def discover_classified_memory_store_plugins(
    *,
    discover_entry_points: bool = True,
    explicit_plugins: Sequence[type] = (),
    on_load_failure: LoadIsolation = "isolate",
) -> tuple[ClassifiedMemoryStorePlugin, ...]:
    """Discover and classify Memory store plugin candidates."""
    classified: list[ClassifiedMemoryStorePlugin] = []

    for plugin_type in explicit_plugins:
        record = classify_memory_store_plugin_record(plugin_type)
        if record is not None:
            classified.append(record)

    if not discover_entry_points:
        return tuple(classified)

    for result in load_entry_point_targets(
        EP_MEMORY_STORES,
        on_load_failure=on_load_failure,
    ):
        if result.error is not None:
            continue
        plugin_type = _plugin_type_from_entry_point_target(result.target, result.spec.value)
        record = classify_memory_store_plugin_record(
            plugin_type,
            entry_point_name=result.spec.name,
            entry_point_spec=result.spec,
        )
        if record is not None:
            classified.append(record)

    return tuple(classified)


def index_classified_memory_store_plugins(
    plugins: Sequence[ClassifiedMemoryStorePlugin],
) -> dict[str, ClassifiedMemoryStorePlugin]:
    """Index classified plugins by ``plugin_id``; duplicate ids fail closed."""
    index: dict[str, ClassifiedMemoryStorePlugin] = {}
    for record in plugins:
        existing = index.get(record.plugin_id)
        if existing is not None:
            raise MemoryStorePluginResolutionError(
                f"Duplicate memory store plugin_id {record.plugin_id!r} "
                f"({existing.entry_point_name!r} vs {record.entry_point_name!r})"
            )
        index[record.plugin_id] = record
    return index


def selected_plugin_load_failed(
    plugin_id: str,
    failed: Sequence[EntryPointLoadResult],
) -> EntryPointLoadResult | None:
    """Return the isolated load failure for ``plugin_id`` when present."""
    for item in failed:
        if item.spec.name == plugin_id:
            return item
    return None
