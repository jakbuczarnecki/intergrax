# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 batch integrity — catalog coverage and evidence collectability."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.catalog import (
    GOV_FINAL_4_FAILURE_CATALOG,
    GOV_FINAL_4_SCENARIO_CATALOG,
    GovFinal4ScenarioResult,
    gov_final_4_failure_evidence_pytest_node_ids,
    gov_final_4_scenario_evidence_pytest_node_ids,
    gov_final_4_unique_catalog_pytest_node_ids,
    scenario_ids_a_through_z,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_are_collectable,
    pytest_nodes_missing_from_collection,
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


def test_gov_final_4_all_catalog_pytest_node_ids_are_collectable() -> None:
    scenario_ids = gov_final_4_scenario_evidence_pytest_node_ids()
    failure_ids = gov_final_4_failure_evidence_pytest_node_ids()
    combined = gov_final_4_unique_catalog_pytest_node_ids()
    missing, proc = pytest_nodes_missing_from_collection(combined, _REPO_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, (
        f"scenario declared={len(scenario_ids)} failure declared={len(failure_ids)} "
        f"unique={len(combined)} missing={len(missing)}: {sorted(missing)}"
    )


def test_gov_final_4_node_validator_rejects_nonexistent_test_in_existing_file() -> None:
    existing_file = gov_final_4_scenario_evidence_pytest_node_ids()[0].split("::", 1)[0]
    fake_node = f"{existing_file}::test_gov_final_4_negative_control_nonexistent_proof"
    assert not pytest_nodes_are_collectable([fake_node], _REPO_ROOT)


def test_gov_final_4_failure_matrix_non_empty() -> None:
    assert len(GOV_FINAL_4_FAILURE_CATALOG) >= 5
