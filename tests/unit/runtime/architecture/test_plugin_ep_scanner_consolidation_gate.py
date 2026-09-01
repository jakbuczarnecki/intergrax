# © Artur Czarnecki. All rights reserved.

"""PLUGIN-EP-SCANNER-CONSOLIDATION-1 — runtime registries must not bypass core discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REGISTRY_SOURCES = (
    Path("intergrax/runtime/nexus/tools/tool_selection_registry.py"),
    Path("intergrax/runtime/execution/budget/registry.py"),
    Path("intergrax/runtime/execution/authority/registry.py"),
)


@pytest.mark.parametrize("source_path", _REGISTRY_SOURCES, ids=[path.name for path in _REGISTRY_SOURCES])
def test_runtime_registry_does_not_use_importlib_entry_points_directly(
    source_path: Path,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    assert "importlib.metadata" not in source
    assert "entry_points" not in source
