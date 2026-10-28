# © Artur Czarnecki. All rights reserved.

"""EBH5-G06 enterprise integrity gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS
from tests.qualification.ebh_5._fixed_substrate_registry import (
    assert_fixed_substrate_closed_world,
    fixed_substrate_decisions,
)
from tests.qualification.ebh_5._inventory_closed_world_validation import (
    assert_inventory_closed_world,
    inventory_validation_errors,
    roadmap_domain_coverage_summary,
    validate_inventory_record,
)
from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY
from tests.qualification.ebh_5._vendor_backend_types import ExtensibilityClassification

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_MD = _REPO_ROOT / "docs/project/maintainers/qualification/EBH_5_CROSS_LAYER_REPLACEABILITY_MATRIX.md"


def test_ebh5_g06_01_inventory_closed_world() -> None:
    assert_inventory_closed_world()
    assert count_unclassified_compat() == 0


def test_ebh5_g06_01_roadmap_31_domains_accounted() -> None:
    summary = roadmap_domain_coverage_summary()
    assert all(not v.endswith("MISSING") for v in summary.values())


def test_ebh5_g06_01_negative_broken_record_rejected() -> None:
    sample = VENDOR_BACKEND_DEPENDENCY_INVENTORY[0]
    broken = replace(sample, semantic_owner="")
    errors = validate_inventory_record(broken)
    assert errors


def test_ebh5_g06_02_fixed_substrate_registry_complete() -> None:
    assert_fixed_substrate_closed_world()
    assert len(fixed_substrate_decisions()) >= 3


def test_ebh5_g06_03_matrix_md_lists_five_qualified() -> None:
    text = _MATRIX_MD.read_text(encoding="utf-8")
    assert "integration.key_value_cache" in text
    for concern_id in QUALIFIED_CONCERN_IDS:
        assert concern_id in text
    matrix_qualified = sum(
        1 for row in CROSS_LAYER_REPLACEABILITY_MATRIX if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
    )
    assert matrix_qualified == len(QUALIFIED_CONCERN_IDS) == 5


def count_unclassified_compat() -> int:
    return len(inventory_validation_errors())
