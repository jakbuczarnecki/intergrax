# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2 adversarial E2E bundle registry parity (E2E-A … E2E-H)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p5_r2_closed_world_adversarial_matrix import (
    P5_CLOSED_WORLD_ADVERSARIAL_MATRIX,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _top_level_test_names(module_path: Path) -> frozenset[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def test_txp5cw_adv01_adversarial_bundle_complete() -> None:
    assert len(P5_CLOSED_WORLD_ADVERSARIAL_MATRIX) == 8
    ids = [row.bundle_id for row in P5_CLOSED_WORLD_ADVERSARIAL_MATRIX]
    assert ids == ["E2E-A", "E2E-B", "E2E-C", "E2E-D", "E2E-E", "E2E-F", "E2E-G", "E2E-H"]


def test_txp5cw_adv02_adversarial_bundle_maps_existing_tests() -> None:
    seen: set[str] = set()
    for row in P5_CLOSED_WORLD_ADVERSARIAL_MATRIX:
        assert row.status == "PASS"
        assert row.test_id not in seen
        seen.add(row.test_id)
        module_path = _REPO_ROOT / row.test_module
        assert module_path.is_file(), row.test_module
        defined = _top_level_test_names(module_path)
        assert row.test_id in defined, (row.test_module, row.test_id)
