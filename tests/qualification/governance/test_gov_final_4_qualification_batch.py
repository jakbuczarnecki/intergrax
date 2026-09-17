# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 batch integrity — catalog coverage and evidence path existence."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.catalog import (
    GOV_FINAL_4_FAILURE_CATALOG,
    GOV_FINAL_4_SCENARIO_CATALOG,
    GovFinal4ScenarioResult,
    scenario_ids_a_through_z,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_gov_final_4_catalog_covers_scenarios_a_through_z() -> None:
    covered = scenario_ids_a_through_z()
    expected = {chr(ord("A") + i) for i in range(26)}
    assert expected <= covered


def test_gov_final_4_catalog_entries_have_evidence_or_explicit_gap() -> None:
    for entry in GOV_FINAL_4_SCENARIO_CATALOG:
        if entry.result is GovFinal4ScenarioResult.GAP:
            assert not entry.primary_pytest_node_ids or entry.notes
            continue
        assert entry.primary_pytest_node_ids, entry.scenario_id


def test_gov_final_4_referenced_test_paths_exist() -> None:
    missing: list[str] = []
    for entry in GOV_FINAL_4_SCENARIO_CATALOG:
        for node_id in entry.primary_pytest_node_ids:
            path = node_id.split("::", 1)[0]
            if not (_REPO_ROOT / path).is_file():
                missing.append(node_id)
    for entry in GOV_FINAL_4_FAILURE_CATALOG:
        for node_id in entry.pytest_node_ids:
            path = node_id.split("::", 1)[0]
            if not (_REPO_ROOT / path).is_file():
                missing.append(node_id)
    assert not missing, f"missing evidence files: {missing}"


def test_gov_final_4_failure_matrix_non_empty() -> None:
    assert len(GOV_FINAL_4_FAILURE_CATALOG) >= 5
