# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P6 adversarial E2E bundle registry parity (P6-A … P6-H)."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p6_adversarial_matrix import P6_ADVERSARIAL_MATRIX
from tests.qualification.trace_x._trace_x_p6_pass1_session import PASS1_OBSERVED_MANIFEST

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _top_level_test_names(module_path: Path) -> frozenset[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )


def test_txp6_adv01_adversarial_bundle_complete() -> None:
    assert len(P6_ADVERSARIAL_MATRIX) == 8
    ids = [row.bundle_id for row in P6_ADVERSARIAL_MATRIX]
    assert ids == [
        "P6-A",
        "P6-B",
        "P6-C",
        "P6-D",
        "P6-E",
        "P6-F",
        "P6-G",
        "P6-H",
    ]


def test_txp6_adv02_adversarial_bundle_maps_existing_tests() -> None:
    seen: set[str] = set()
    for row in P6_ADVERSARIAL_MATRIX:
        assert row.status == "PASS"
        assert row.test_id not in seen
        seen.add(row.test_id)
        module_path = _REPO_ROOT / row.test_module
        assert module_path.is_file(), row.test_module
        defined = _top_level_test_names(module_path)
        assert row.test_id in defined, (row.test_module, row.test_id)


def _nodeid_passed_in_manifest(full_nodeid: str, passed: set[str]) -> bool:
    from tests.qualification.trace_x._trace_x_p6_pass1_session import _test_id_observed

    test_name = full_nodeid.replace("\\", "/").rsplit("::", 1)[-1].split("[", 1)[0]
    return _test_id_observed(passed, test_name)


def test_txp6_adv03_adversarial_bundle_passed_in_qualification_session() -> None:
    if os.environ.get("TRACE_X_P6_PASS1") == "1":
        return
    if not PASS1_OBSERVED_MANIFEST.is_file():
        pytest.skip(
            "adversarial bundle session manifest missing; rerun qualification batch with "
            "TRACE_X_P6_PASS1=1 including all P6-A..H tests",
        )
    passed = set(json.loads(PASS1_OBSERVED_MANIFEST.read_text(encoding="utf-8")))
    missing = [
        f"{row.test_module}::{row.test_id}"
        for row in P6_ADVERSARIAL_MATRIX
        if not _nodeid_passed_in_manifest(f"{row.test_module}::{row.test_id}", passed)
    ]
    assert not missing, missing
