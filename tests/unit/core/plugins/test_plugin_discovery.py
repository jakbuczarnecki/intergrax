# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.plugins.discovery import load_plugin_types
from intergrax.core.plugins.errors import PluginLoadError
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin

pytestmark = pytest.mark.unit


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
