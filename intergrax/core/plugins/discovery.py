# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Load Tier-0 catalog plugins from setuptools entry points."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, TypeVar

from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError

logger = logging.getLogger(__name__)

ConflictPolicy = Literal["error", "skip", "override", "warn_override"]

EP_INTEGRATIONS = "intergrax.integrations"
EP_TOOLS = "intergrax.tools"
EP_SKILLS = "intergrax.skills"

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    """One resolved entry-point plugin."""

    name: str
    group: str
    plugin_type: type


def _iter_entry_point_specs(group: str) -> list[tuple[str, str]]:
    try:
        from importlib.metadata import entry_points
    except ImportError as exc:  # pragma: no cover
        raise PluginLoadError("importlib.metadata unavailable") from exc

    eps = entry_points()
    selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])
    specs: list[tuple[str, str]] = []
    for ep in selected:
        name = getattr(ep, "name", "")
        value = getattr(ep, "value", "")
        if name and value:
            specs.append((name, value))
    return specs


def _load_target(value: str) -> type:
    module_path, _, attr = value.partition(":")
    if not module_path or not attr:
        raise PluginLoadError(f"Invalid entry point target {value!r}; expected 'module:attr'")
    module = importlib.import_module(module_path)
    target = getattr(module, attr)
    if isinstance(target, type):
        return target
    if callable(target):
        loaded = target()
        if not isinstance(loaded, type):
            raise PluginLoadError(f"Factory {value!r} must return a plugin class")
        return loaded
    raise PluginLoadError(f"Entry point {value!r} is not a class or factory")


def load_entry_point_plugins(
    group: str,
    *,
    on_conflict: ConflictPolicy = "error",
    seen: set[str] | None = None,
) -> list[LoadedPlugin]:
    """
    Discover and import plugin classes for ``group``.

    ``on_conflict`` applies to duplicate entry point ``name`` values only.
    """
    known = seen if seen is not None else set()
    loaded: list[LoadedPlugin] = []
    for name, value in _iter_entry_point_specs(group):
        if name in known:
            if on_conflict == "skip":
                logger.warning("Skipping duplicate entry point %s in group %s", name, group)
                continue
            if on_conflict in ("override", "warn_override"):
                logger.warning("Overriding duplicate entry point %s in group %s", name, group)
            else:
                raise PluginConflictError(
                    f"Duplicate entry point {name!r} in group {group!r}",
                    plugin_name=name,
                    group=group,
                )
        known.add(name)
        try:
            plugin_type = _load_target(value)
        except Exception as exc:
            raise PluginLoadError(f"Failed to load {group}:{name} ({value}): {exc}") from exc
        loaded.append(LoadedPlugin(name=name, group=group, plugin_type=plugin_type))
    return loaded


def load_plugin_types(
    group: str,
    *,
    explicit: Sequence[type] = (),
    discover_entry_points: bool = False,
    on_conflict: ConflictPolicy = "error",
) -> list[type]:
    """Merge explicit plugin classes with optional entry-point discovery."""
    types: list[type] = []
    seen_names: set[str] = set()
    for plugin_type in explicit:
        types.append(plugin_type)
    if discover_entry_points:
        for item in load_entry_point_plugins(group, on_conflict=on_conflict, seen=seen_names):
            types.append(item.plugin_type)
    return types


def register_plugins(
    group: str,
    register_fn: Callable[[type], bool | None],
    *,
    explicit: Sequence[type] = (),
    discover_entry_points: bool = False,
    on_conflict: ConflictPolicy = "error",
) -> int:
    """Load plugins and invoke ``register_fn(plugin_type)`` for each.

    When ``register_fn`` returns ``False``, the plugin is not counted (e.g. catalog skip).
    """
    count = 0
    for plugin_type in load_plugin_types(
        group,
        explicit=explicit,
        discover_entry_points=discover_entry_points,
        on_conflict=on_conflict,
    ):
        registered = register_fn(plugin_type)
        if registered is not False:
            count += 1
    return count
