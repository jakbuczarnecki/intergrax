# © Artur Czarnecki. All rights reserved.

"""GR-10 batch — catalog integrity and mapped evidence collectability."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_are_collectable,
    pytest_nodes_missing_from_collection,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_SCENARIO_CATALOG,
    gr10_catalog_pytest_node_ids,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_gr10_scenario_catalog_non_empty() -> None:
    assert len(GR10_SCENARIO_CATALOG) >= 10


def test_gr10_all_catalog_pytest_node_ids_are_collectable() -> None:
    node_ids = gr10_catalog_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(node_ids, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, sorted(missing)


def test_gr10_node_validator_rejects_fake_node() -> None:
    existing = gr10_catalog_pytest_node_ids()[0]
    file_part = existing.split("::", 1)[0]
    fake = f"{file_part}::test_gr10_negative_control_nonexistent"
    assert not pytest_nodes_are_collectable([fake], _REPO_ROOT)
