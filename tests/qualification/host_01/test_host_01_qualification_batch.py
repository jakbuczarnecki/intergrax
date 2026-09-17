# © Artur Czarnecki. All rights reserved.

"""HOST-01 batch hooks and catalog integrity."""

from __future__ import annotations

import pytest

from tests.qualification.host_01.catalog import HOST_01_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_host_01_catalog_covers_host_q1_through_host_q12() -> None:
    ids = {entry.q_id for entry in HOST_01_Q_CATALOG}
    expected = {f"HOST-Q{i}" for i in range(1, 13)}
    assert ids == expected


def test_host_01_frozen_host_convergence_regression_paths() -> None:
    paths = (
        "tests/unit/runtime/architecture/test_platform_execution_unification_u1_application_scenario_entry.py",
        "tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py",
        "tests/unit/runtime/architecture/test_ue_11gp_production_host_execution_gate.py",
        "tests/unit/applications/test_mcp_canonical_execution.py",
    )
    for path in paths:
        assert path.startswith("tests/")
