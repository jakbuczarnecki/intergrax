# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.core.plugins.discovery import load_entry_point_plugins, load_plugin_types
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


def test_load_plugin_types_invalid_target_raises() -> None:
    with pytest.raises(PluginLoadError, match="Invalid entry point"):
        from intergrax.core.plugins.discovery import _load_target

        _load_target("not-a-valid-target")


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
