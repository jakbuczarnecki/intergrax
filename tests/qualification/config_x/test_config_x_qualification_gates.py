# © Artur Czarnecki. All rights reserved.

"""CONFIG-X closed-world qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory

from tests.qualification.config_x._config_x_blockers import (
    CONFIG_X_BLOCKER_RECORDS,
    blocker_counts_by_classification,
)
from tests.qualification.config_x._config_x_concern_inventory import (
    CONFIG_X_CONCERN_INVENTORY,
    config_x_concern_inventory,
)
from tests.qualification.config_x._config_x_discovery import (
    classify_synthetic_unknown_surface,
    discover_blocker_path_keys,
    discover_composition_root_paths,
    discover_integration_category_enum_size,
)
from tests.qualification.config_x._config_x_owner_discovery import (
    CONFIG_X_OWNER_EXPECTATIONS,
    compare_owner_gate,
)
from tests.qualification.config_x._config_x_types import (
    BLOCKER_CLASSIFICATIONS,
    ConfigClassification,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_cx_q01_concern_inventory_closed_world() -> None:
    integration_count = len(tuple(IntegrationCategory))
    platform_count = len(CONFIG_X_CONCERN_INVENTORY) - integration_count
    assert integration_count == discover_integration_category_enum_size()
    assert integration_count >= 30
    assert platform_count >= 15
    concern_ids = [row.concern_id for row in CONFIG_X_CONCERN_INVENTORY]
    assert len(concern_ids) == len(set(concern_ids))
    for row in CONFIG_X_CONCERN_INVENTORY:
        assert row.classification not in BLOCKER_CLASSIFICATIONS


def test_cx_q02_concern_inventory_matches_integration_categories() -> None:
    expected = {f"integration.{cat.value}" for cat in IntegrationCategory}
    actual = {
        row.concern_id
        for row in CONFIG_X_CONCERN_INVENTORY
        if row.concern_id.startswith("integration.")
    }
    assert actual == expected


def test_cx_q03_composition_roots_discovered_subset_of_production_tree() -> None:
    discovered = discover_composition_root_paths()
    assert len(discovered) >= 20
    for path in discovered:
        assert (_REPO_ROOT / path).is_file()


def test_cx_q04_blocker_paths_exist_on_disk() -> None:
    for row in CONFIG_X_BLOCKER_RECORDS:
        for path in row.paths:
            assert (_REPO_ROOT / path).is_file(), path


def test_cx_q05_blocker_inventory_parity() -> None:
    discovered = discover_blocker_path_keys()
    expected: set[str] = set()
    for row in CONFIG_X_BLOCKER_RECORDS:
        expected.update(row.paths)
    assert discovered == expected


def test_cx_q06_owner_discovery_integration_provider_selection() -> None:
    discovered, expected = compare_owner_gate("integration_provider_selection")
    assert discovered == expected


def test_cx_q07_owner_discovery_llm_provider_selection() -> None:
    discovered, expected = compare_owner_gate("llm_provider_selection")
    assert discovered == expected


def test_cx_q08_owner_discovery_execution_bound_integration() -> None:
    discovered, expected = compare_owner_gate("execution_bound_integration_resolution")
    assert discovered == expected


def test_cx_q09_owner_discovery_existing_capability_realization() -> None:
    discovered, expected = compare_owner_gate("existing_capability_configuration_realization")
    assert discovered == expected


def test_cx_q10_owner_discovery_plugin_catalog() -> None:
    discovered, expected = compare_owner_gate("plugin_integration_catalog")
    assert discovered == expected


def test_cx_q11_owner_matrix_covers_mandatory_concerns() -> None:
    assert set(CONFIG_X_OWNER_EXPECTATIONS) >= {
        "integration_provider_selection",
        "llm_provider_selection",
        "execution_bound_integration_resolution",
        "existing_capability_configuration_realization",
        "plugin_integration_catalog",
    }


def test_cx_q12_classification_sensitivity_synthetic_unclassified_fails_closed() -> None:
    assert classify_synthetic_unknown_surface("# CONFIG_X_SYNTHETIC_UNCLASSIFIED\n") is None
    assert classify_synthetic_unknown_surface("normal module") == "H"


def test_cx_q13_inventory_reload_stable() -> None:
    assert config_x_concern_inventory() == CONFIG_X_CONCERN_INVENTORY


def test_cx_q14_blocker_exit_counts_documented() -> None:
    counts = blocker_counts_by_classification()
    assert counts[ConfigClassification.I_HARD_CODED_PRODUCTION_SELECTION] == 3
    assert counts[ConfigClassification.J_SILENT_FALLBACK] == 2
    assert counts[ConfigClassification.K_DUPLICATE_CONFIGURATION_AUTHORITY] == 0
    assert counts[ConfigClassification.L_UNCLEAR] == 0
    assert len(CONFIG_X_BLOCKER_RECORDS) == 5
