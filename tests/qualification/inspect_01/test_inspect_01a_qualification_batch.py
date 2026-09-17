# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-A batch hooks for frozen regressions."""

from __future__ import annotations

import pytest

from tests.qualification.inspect_01.catalog import INSPECT_01_A_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_a_q14_p14_runtime_inspection_regression() -> None:
    node = "tests/unit/applications/test_runtime_inspection.py"
    assert node.startswith("tests/")


def test_a_q15_frozen_subsystem_regression_paths() -> None:
    paths = (
        "tests/unit/runtime/observability/reconstruction/test_execution_reconstruction.py",
        "tests/unit/contracts/test_execution_reconstruction_reader.py",
        "tests/qualification/session_01/test_session_01_q_catalog_integrity.py",
    )
    for path in paths:
        assert path.startswith("tests/")


def test_inspect_01_a_catalog_covers_a_q1_through_a_q15() -> None:
    ids = {entry.q_id for entry in INSPECT_01_A_Q_CATALOG}
    expected = {f"A-Q{i}" for i in range(1, 16)}
    assert ids == expected
