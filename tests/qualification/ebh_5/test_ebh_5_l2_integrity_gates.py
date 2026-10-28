# © Artur Czarnecki. All rights reserved.

"""EBH-5 L2 integrity gates — inventory, matrix, CONFIG-X reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.integrations.contracts.base import IntegrationCategory
from tests.qualification.config_x._config_x_concern_inventory import CONFIG_X_CONCERN_INVENTORY
from tests.qualification.ebh_5._closed_world_discovery import (
    discover_composition_root_paths,
    discover_integration_category_enum_size,
    discover_sqlite_connect_paths_under_intergrax,
)
from tests.qualification.ebh_5._config_x_reconciliation import (
    CONFIG_X_CONCERN_COUNT,
    assert_config_x_coverage_complete,
    config_x_reconciliation_rows,
)
from tests.qualification.ebh_5._evidence_manifest import EBH5_EVIDENCE_SUITE, evidence_git_head_sha
from tests.qualification.ebh_5._replaceability_matrix import (
    CROSS_LAYER_REPLACEABILITY_MATRIX,
    undocumented_qualified_rows,
)
from tests.qualification.ebh_5._vendor_backend_inventory import (
    VENDOR_BACKEND_DEPENDENCY_INVENTORY,
    count_unclassified,
    inventory_concern_ids,
    vendor_backend_concern_inventory,
)
from tests.qualification.ebh_5._vendor_backend_types import (
    ExtensibilityPromised,
    RoadmapVendorDomain,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

MANDATORY_ROADMAP_DOMAINS = frozenset(RoadmapVendorDomain)


def test_ebh5_q01_inventory_unique_ids() -> None:
    ids = [row.concern_id for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY]
    assert len(ids) == len(set(ids))
    assert len(VENDOR_BACKEND_DEPENDENCY_INVENTORY) >= CONFIG_X_CONCERN_COUNT + 10


def test_ebh5_q02_config_x_concerns_mapped() -> None:
    lineage = frozenset(
        row.concern_id for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY if row.config_x_concern_id is not None
    )
    assert_config_x_coverage_complete(lineage)
    assert len(config_x_reconciliation_rows()) == CONFIG_X_CONCERN_COUNT


def test_ebh5_q03_unclassified_zero() -> None:
    assert count_unclassified() == 0


def test_ebh5_q04_all_records_have_owner_and_domain() -> None:
    for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY:
        assert row.semantic_owner.strip()
        assert row.composition_selection_owner.strip()
        assert row.domain in MANDATORY_ROADMAP_DOMAINS


def test_ebh5_q05_integration_categories_covered() -> None:
    expected = {f"integration.{cat.value}" for cat in IntegrationCategory}
    actual = {cid for cid in inventory_concern_ids() if cid.startswith("integration.")}
    assert actual == expected
    assert discover_integration_category_enum_size() == len(tuple(IntegrationCategory))


def test_ebh5_q06_matrix_covers_extensibility_promised() -> None:
    promised_ids = {
        row.concern_id
        for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY
        if row.extensibility_promised in (ExtensibilityPromised.YES, ExtensibilityPromised.PARTIAL)
    }
    matrix_ids = {row.concern_id for row in CROSS_LAYER_REPLACEABILITY_MATRIX}
    missing = promised_ids - matrix_ids
    assert not missing, sorted(missing)


def test_ebh5_q07_no_undocumented_qualified() -> None:
    assert undocumented_qualified_rows() == ()


def test_ebh5_q08_composition_roots_exist() -> None:
    for path in discover_composition_root_paths():
        assert (_REPO_ROOT / path).is_file(), path


def test_ebh5_q09_sqlite_annex_candidates_inventoried() -> None:
    sqlite_paths = discover_sqlite_connect_paths_under_intergrax()
    assert len(sqlite_paths) >= 30
    inventory_paths = frozenset(row.vendor_code_location for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY)
    covered = sum(
        1
        for p in sqlite_paths
        if any(p.startswith(prefix.rstrip("*")) or prefix in p for prefix in inventory_paths)
        or any(p in row.vendor_code_location for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY)
    )
    assert covered >= 10


def test_ebh5_q10_inventory_reload_stable() -> None:
    assert vendor_backend_concern_inventory() == VENDOR_BACKEND_DEPENDENCY_INVENTORY


def test_ebh5_q11_config_x_inventory_unchanged_count() -> None:
    assert len(CONFIG_X_CONCERN_INVENTORY) == CONFIG_X_CONCERN_COUNT == 54


def test_ebh5_q12_evidence_suite_and_sha() -> None:
    sha = evidence_git_head_sha()
    assert len(sha) == 40
    assert (_REPO_ROOT / EBH5_EVIDENCE_SUITE).is_dir()
