# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Load Tier-0 catalog plugins from setuptools entry points."""

from __future__ import annotations
from intergrax.utils import attribute_access

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Sequence, TypeVar

from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError
from intergrax.core.plugins.platform_semantics import PlatformPluginConflictKind

logger = logging.getLogger(__name__)

ConflictPolicy = Literal["error", "skip", "override", "warn_override"]
LoadIsolation = Literal["fail_fast", "isolate"]

EP_INTEGRATIONS = "intergrax.integrations"
EP_TOOLS = "intergrax.tools"
EP_SKILLS = "intergrax.skills"
EP_MEMORY_STORES = "intergrax.memory_stores"
EP_CONTEXT = "intergrax.context"
EP_RAG_CHUNKERS = "intergrax.rag.chunkers"
EP_RAG_RETRIEVERS = "intergrax.rag.retrievers"
EP_RAG_RERANKERS = "intergrax.rag.rerankers"
EP_SECURITY_DEFENSES = "intergrax.security_defenses"
EP_POLICY_RULES = "intergrax.policy_rules"
EP_TOOL_INVOCATION_PATTERNS = "intergrax.tool_invocation_patterns"

T = TypeVar("T")

_EP_SPECS_CACHE: dict[str, tuple[EntryPointSpec, ...]] = {}
_EP_SPECS_BY_NAME: dict[str, dict[str, EntryPointSpec]] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class EntryPointSpec:
    """One setuptools entry point in ``group`` without loading its target."""

    name: str
    group: str
    value: str
    distribution: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedPlugin:
    """One resolved entry-point plugin."""

    name: str
    group: str
    plugin_type: type


@dataclass(frozen=True, slots=True)
class EntryPointLoadResult:
    """Loaded entry-point target or isolated load failure."""

    spec: EntryPointSpec
    target: object | None = None
    error: BaseException | None = None


def _scan_entry_point_specs(group: str) -> tuple[EntryPointSpec, ...]:
    """Enumerate entry points for ``group`` without caching (internal)."""
    try:
        from importlib.metadata import entry_points
    except ImportError as exc:  # pragma: no cover
        raise PluginLoadError("importlib.metadata unavailable") from exc

    eps = entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        selected = eps.get(group, [])  # type: ignore[union-attr]
    specs: list[EntryPointSpec] = []
    for ep in selected:
        name = str(attribute_access.optional(ep, "name", ""))
        value = str(attribute_access.optional(ep, "value", ""))
        if not name or not value:
            continue
        dist = attribute_access.optional(ep, "dist", None)
        distribution: str | None = None
        if dist is not None:
            raw_name = attribute_access.optional(dist, "name", None)
            if isinstance(raw_name, str):
                distribution = raw_name
        specs.append(
            EntryPointSpec(
                name=name,
                group=group,
                value=value,
                distribution=distribution,
            )
        )
    return tuple(sorted(specs, key=lambda item: (item.name, item.value)))


def iter_entry_point_specs(group: str) -> tuple[EntryPointSpec, ...]:
    """Enumerate entry points for ``group`` in deterministic order (scan only, cached)."""
    with _CACHE_LOCK:
        cached = _EP_SPECS_CACHE.get(group)
        if cached is not None:
            return cached
    specs = _scan_entry_point_specs(group)
    with _CACHE_LOCK:
        existing = _EP_SPECS_CACHE.get(group)
        if existing is not None:
            return existing
        _EP_SPECS_CACHE[group] = specs
        by_name: dict[str, EntryPointSpec] = {}
        for spec in specs:
            if spec.name not in by_name:
                by_name[spec.name] = spec
        _EP_SPECS_BY_NAME[group] = by_name
        return specs


def get_entry_point_spec(group: str, name: str) -> EntryPointSpec | None:
    """Return one cached entry-point spec by ``name``, or ``None`` when absent."""
    iter_entry_point_specs(group)
    with _CACHE_LOCK:
        return _EP_SPECS_BY_NAME.get(group, {}).get(name)


def reset_entry_point_spec_cache_for_tests() -> None:
    """Clear the per-process entry-point spec cache (tests and dev bootstrap only)."""
    with _CACHE_LOCK:
        _EP_SPECS_CACHE.clear()
        _EP_SPECS_BY_NAME.clear()


def load_entry_point_value(value: str) -> object:
    """Load an entry-point target without domain registration or factory invocation.

    Semantics match ``importlib.metadata.EntryPoint.load()``: classes, functions,
    and callable objects are returned without instantiation or execution.
    """
    try:
        from importlib.metadata import EntryPoint
    except ImportError as exc:  # pragma: no cover
        raise PluginLoadError("importlib.metadata unavailable") from exc
    try:
        return EntryPoint(name="", group="", value=value).load()
    except Exception as exc:
        raise PluginLoadError(f"Failed to load entry point target {value!r}: {exc}") from exc


def instantiate_entry_point_target(target: object) -> object:
    """Instantiate ``target`` when it is a class; return instances unchanged."""
    if isinstance(target, type):
        return target()
    return target


def load_entry_point_targets(
    group: str,
    *,
    on_conflict: ConflictPolicy = "error",
    on_load_failure: LoadIsolation = "fail_fast",
    seen: set[str] | None = None,
) -> list[EntryPointLoadResult]:
    """Load entry-point targets for ``group`` without domain registration."""
    known = seen if seen is not None else set()
    loaded: list[EntryPointLoadResult] = []
    for spec in iter_entry_point_specs(group):
        if spec.name in known:
            if on_conflict == "skip":
                logger.warning("Skipping duplicate entry point %s in group %s", spec.name, group)
                continue
            if on_conflict in ("override", "warn_override"):
                logger.warning("Overriding duplicate entry point %s in group %s", spec.name, group)
            else:
                raise PluginConflictError(
                    f"Duplicate entry point {spec.name!r} in group {group!r}",
                    plugin_name=spec.name,
                    group=group,
                    conflict_kind=PlatformPluginConflictKind.ENTRY_POINT_NAME,
                )
        known.add(spec.name)
        try:
            target = load_entry_point_value(spec.value)
        except PluginLoadError as exc:
            if on_load_failure == "isolate":
                loaded.append(EntryPointLoadResult(spec=spec, error=exc))
                continue
            raise PluginLoadError(
                f"Failed to load {group}:{spec.name} ({spec.value}): {exc}"
            ) from exc
        loaded.append(EntryPointLoadResult(spec=spec, target=target))
    return loaded


def _resolve_tier0_plugin_type(target: object, value: str) -> type:
    """Resolve a Tier-0 plugin class, invoking callable factories when needed."""
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


def _load_target(value: str) -> type:
    return _resolve_tier0_plugin_type(load_entry_point_value(value), value)


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
    loaded: list[LoadedPlugin] = []
    for result in load_entry_point_targets(
        group,
        on_conflict=on_conflict,
        seen=seen,
    ):
        if result.error is not None:
            raise result.error
        plugin_type = _resolve_tier0_plugin_type(result.target, result.spec.value)
        loaded.append(
            LoadedPlugin(
                name=result.spec.name,
                group=group,
                plugin_type=plugin_type,
            )
        )
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
