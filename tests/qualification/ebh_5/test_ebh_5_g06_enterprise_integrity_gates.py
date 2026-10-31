# © Artur Czarnecki. All rights reserved.

"""EBH5-G06 enterprise integrity gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.qualification.ebh_5._behavioral_proof_registry import QUALIFIED_CONCERN_IDS
from tests.qualification.ebh_5._fixed_substrate_registry import (
    FixedSubstrateDecisionStatus,
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
from tests.qualification.ebh_5._roadmap_domain_na_evidence import roadmap_domain_na_validation_errors
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


def test_ebh5_g06_01_negative_broken_proof_reference_rejected() -> None:
    sample = next(r for r in VENDOR_BACKEND_DEPENDENCY_INVENTORY if r.concern_id == "integration.document_store")
    broken = replace(
        sample,
        extensibility_classification=ExtensibilityClassification.REPLACEABLE_NOT_YET,
        behavioral_qualification_evidence_id="bogus-proof-ref",
        outstanding_debt="synthetic negative probe",
    )
    assert validate_inventory_record(broken)


def test_ebh5_g06_01_negative_broken_record_rejected() -> None:
    sample = VENDOR_BACKEND_DEPENDENCY_INVENTORY[0]
    broken = replace(sample, semantic_owner="")
    errors = validate_inventory_record(broken)
    assert errors


def test_ebh5_g06_01_roadmap_six_na_domains_have_discovery_evidence() -> None:
    assert not roadmap_domain_na_validation_errors()


def test_ebh5_g06_02_fixed_substrate_ratification_status_explicit() -> None:
    for row in fixed_substrate_decisions():
        assert row.decision_status == FixedSubstrateDecisionStatus.RATIFIED
    assert_fixed_substrate_closed_world()
    assert len(fixed_substrate_decisions()) >= 3


def test_ebh5_g06_03_matrix_md_lists_qualified_at_head() -> None:
    text = _MATRIX_MD.read_text(encoding="utf-8")
    assert "integration.key_value_cache" in text
    matrix_qualified = sum(
        1 for row in CROSS_LAYER_REPLACEABILITY_MATRIX if row.rpl_h == ExtensibilityClassification.REPLACEABLE_QUALIFIED
    )
    assert matrix_qualified == len(QUALIFIED_CONCERN_IDS) == 56


def test_ebh5_g06_04_evidence_manifest_head_matches_git() -> None:
    from tests.qualification.ebh_5._evidence_manifest import certification_handoff_sha, evidence_git_head_sha

    assert certification_handoff_sha() == evidence_git_head_sha()


def count_unclassified_compat() -> int:
    return len(inventory_validation_errors())
