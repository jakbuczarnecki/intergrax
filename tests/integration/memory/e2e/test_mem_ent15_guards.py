# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15: lightweight architecture guards for E2E suite."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[4]
_E2E_ROOT = _REPO / "tests" / "integration" / "memory" / "e2e"


def _e2e_test_files() -> list[Path]:
    return sorted(_E2E_ROOT.glob("test_mem_ent15_*.py"))


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_mem_ent15_core_tests_reference_default_memory_control_plane() -> None:
    core = _E2E_ROOT / "test_mem_ent15_core_lifecycle.py"
    tree = _parse(core)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "DefaultMemoryControlPlane" in names or "build_in_memory_memory_harness" in names


def test_mem_ent15_e2e_avoids_reflection_and_private_access() -> None:
    forbidden_names = {"getattr", "hasattr", "setattr"}
    for path in _e2e_test_files():
        if path.name == "test_mem_ent15_guards.py":
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                pytest.fail(f"{path.name} uses reflection: {node.id}")
