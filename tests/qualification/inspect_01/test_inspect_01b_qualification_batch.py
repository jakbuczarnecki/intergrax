# © Artur Czarnecki. All rights reserved.

"""INSPECT-01-B batch hooks for regressions."""

from __future__ import annotations

import pytest

from tests.qualification.inspect_01.catalog import INSPECT_01_A_Q_CATALOG, INSPECT_01_B_Q_CATALOG

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_b_q19_inspect_01_a_regression_paths() -> None:
    for entry in INSPECT_01_A_Q_CATALOG:
        assert entry.pytest_node_ids


def test_b_q20_frozen_subsystem_regression_paths() -> None:
    paths = (
        "tests/unit/runtime/nexus/tools/test_tool_runtime_authority_closure.py",
        "tests/unit/runtime/agent_governance/test_agent_runtime_governance.py",
        "tests/qualification/session_01/test_session_01_q_catalog_integrity.py",
        "tests/unit/contracts/test_invoke_tool_runtime_events.py",
    )
    for path in paths:
        assert path.startswith("tests/")


def test_inspect_01_b_catalog_covers_b_q1_through_b_q20() -> None:
    ids = {entry.q_id for entry in INSPECT_01_B_Q_CATALOG}
    expected = {f"B-Q{i}" for i in range(1, 21)}
    assert ids == expected
