# © Artur Czarnecki. All rights reserved.

"""EE-B2-FINAL — fault matrix F-01..F-12 coverage registry."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_b2_final_facts import (
    FAULT_MATRIX_ROWS,
    _REPO_ROOT,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b2_final_fault_matrix_has_twelve_rows() -> None:
    assert len(FAULT_MATRIX_ROWS) == 12
    ids = [row[0] for row in FAULT_MATRIX_ROWS]
    assert ids == [f"F-{i:02d}" for i in range(1, 13)]


def test_ee_b2_final_fault_matrix_modules_exist() -> None:
    missing: list[str] = []
    for fault_id, _inj, _owner, _contain, module in FAULT_MATRIX_ROWS:
        path = _REPO_ROOT / "tests" / "unit" / "runtime" / "architecture" / module
        if not path.is_file():
            missing.append(f"{fault_id}:{module}")
    assert missing == []


def test_ee_b2_final_fault_matrix_no_duplicate_fault_ids() -> None:
    ids = [row[0] for row in FAULT_MATRIX_ROWS]
    assert len(ids) == len(set(ids))
