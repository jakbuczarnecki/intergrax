# © Artur Czarnecki. All rights reserved.

"""HARDENING-1 — gate modules that previously blocked ``tests/unit`` collection."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

_DISCOVERY_BLOCKING_TEST_MODULES: tuple[str, ...] = (
    "tests/unit/runtime/hooks/test_tool_and_selection_hooks.py",
    "tests/unit/applications/test_reference_apps_mcp.py",
    "tests/unit/applications/test_mcp_surface_opt_in.py",
    "tests/unit/applications/test_mcp_surface_opt_in_runtime.py",
    "tests/unit/applications/local_workspace_application/test_lkw_windows_interaction_proof.py",
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.parametrize("relative_path", _DISCOVERY_BLOCKING_TEST_MODULES)
def test_hardening_1_discovery_blocking_test_module_parses(relative_path: str) -> None:
    path = _REPO_ROOT / relative_path
    ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
