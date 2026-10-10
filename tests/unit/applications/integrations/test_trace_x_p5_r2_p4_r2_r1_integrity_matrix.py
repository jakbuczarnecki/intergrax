# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R2-R1-R1-R1 P4 integrity qualification matrix (blocker 34)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.applications.integrations._p4_integrity_matrix_catalog import (
    P4_INTEGRITY_QUALIFICATION_MATRIX,
)
pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _top_level_test_function_names(module_path: Path) -> frozenset[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def test_p4_integrity_matrix_registry_maps_existing_tests() -> None:
    assert len(P4_INTEGRITY_QUALIFICATION_MATRIX) >= 20
    seen_ids: set[str] = set()
    for row in P4_INTEGRITY_QUALIFICATION_MATRIX:
        assert row.scenario.strip()
        assert row.expected.strip()
        assert row.test_module.endswith(".py")
        assert row.test_id.startswith("test_")
        assert row.status == "PASS"
        assert row.test_id not in seen_ids, row.test_id
        seen_ids.add(row.test_id)
        module_path = _REPO_ROOT / row.test_module
        assert module_path.is_file(), row.test_module
        defined = _top_level_test_function_names(module_path)
        assert row.test_id in defined, (row.test_module, row.test_id)
