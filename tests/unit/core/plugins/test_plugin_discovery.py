# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.discovery import (
    instantiate_entry_point_target,
    iter_entry_point_specs,
    load_entry_point_plugins,
    load_entry_point_targets,
    load_entry_point_value,
    load_plugin_types,
)
from intergrax.core.plugins.errors import PluginConflictError, PluginLoadError
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin

pytestmark = pytest.mark.unit


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _DiscoveredPlugin:
    pass


def test_load_plugin_types_explicit_only() -> None:
    types = load_plugin_types(
        "intergrax.integrations",
        explicit=(CustomMemoryKvPlugin,),
        discover_entry_points=False,
    )
    assert types == [CustomMemoryKvPlugin]


def test_load_entry_point_value_invalid_target_raises() -> None:
    with pytest.raises(PluginLoadError, match="Invalid entry point target"):
        load_entry_point_value("not-a-valid-target")


def test_iter_entry_point_specs_is_deterministic_and_scan_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("b", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("a", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    specs = iter_entry_point_specs("intergrax.rag.chunkers")

    assert [spec.name for spec in specs] == ["a", "b"]
    assert all(spec.group == "intergrax.rag.chunkers" for spec in specs)


def test_load_entry_point_targets_isolates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("good", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("bad", "not-a-valid-target", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    results = load_entry_point_targets(
        "intergrax.rag.chunkers",
        on_load_failure="isolate",
    )

    assert len(results) == 2
    assert results[0].spec.name == "bad"
    assert isinstance(results[0].error, PluginLoadError)
    assert results[1].spec.name == "good"
    assert results[1].target is _DiscoveredPlugin


def test_instantiate_entry_point_target_instantiates_classes() -> None:
    instance = instantiate_entry_point_target(_DiscoveredPlugin)
    assert isinstance(instance, _DiscoveredPlugin)


def test_load_entry_point_plugins_scan_does_not_register_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("chunker", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers")]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    specs = iter_entry_point_specs("intergrax.rag.chunkers")
    loaded = load_entry_point_plugins("intergrax.rag.chunkers")

    assert len(specs) == 1
    assert loaded[0].plugin_type is _DiscoveredPlugin


def test_load_entry_point_plugins_selects_requested_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "chunker",
                f"{__name__}:_DiscoveredPlugin",
                "intergrax.rag.chunkers",
            ),
            _EntryPoint(
                "other",
                f"{__name__}:_DiscoveredPlugin",
                "intergrax.rag.retrievers",
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    loaded = load_entry_point_plugins("intergrax.rag.chunkers")

    assert [(item.name, item.group, item.plugin_type) for item in loaded] == [
        ("chunker", "intergrax.rag.chunkers", _DiscoveredPlugin)
    ]


def test_load_entry_point_plugins_invalid_target_raises_canonical_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [_EntryPoint("broken", "not-a-valid-target", "intergrax.rag.chunkers")]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(PluginLoadError, match="Invalid entry point target"):
        load_entry_point_plugins("intergrax.rag.chunkers")


def test_load_entry_point_plugins_rejects_duplicate_external_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.rag.chunkers"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(PluginConflictError, match="Duplicate entry point"):
        load_entry_point_plugins("intergrax.rag.chunkers")
